import os.path as osp
import numpy as np
import json
import glob
import enum
import torch
import math
import argparse
from tqdm import tqdm
from tabulate import tabulate
from multiprocessing import Pool
from sklearn.cluster import KMeans

from mmcv.ops import diff_iou_rotated_3d
from mmdet3d.datasets import Det3DDataset

# Debugging
from joblib import Memory
memory = Memory(location=".joblib_cache", verbose=0)


def humanize(n: float) -> str:
    """
        Make large number more readable for humans.
        M for million and K for thousand.
        Example: 2300 -> 2.3k

        Args:
            n (float):  number to convert
    """
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


class AnchorMode(enum.Enum):
    """
        Enum to controll anchor generation.

        Args:
            PER_CLASS (0)   - generate anchor per class
            PER_DATASET (1) - generate anchor using whole dataset
    """
    PER_CLASS = 0
    PER_DATASET = 1

    def __str__(self):
        return self.name


class AnalysisMode(enum.Enum):
    """
        Enum to control different analysis mode to run.

        Args:
            ALL (-1)        - run all modes below in order
            ANCHORS (0)     - run only anchor generation
            PCD_RANGE (1)   - run only point analysis
            CLASSES (2)     - run only class and bbox analysis
    """
    ALL = -1
    ANCHORS = 0
    PCD_RANGE = 1
    CLASSES = 2

    def __str__(self):
        return self.name


class DatasetAnalyzer:
    """
    Generic Dataset Analyzer for MMDetection3D datasets for statistics like:
        - anchor sizes (per class or per whole dataset)
        - point cloud range, voxel size and SECOND output shape
        - class balance statistics and bbox statistics

    Inherit this class and adapt to dataset.

    Args:
        dataset (Det3DDataset):     MMDetection3D dataset to analyze
        labels_path (str):          path to converted annotations
        points_path (str):          path to converted point clouds
        modes (AnalysisMode):       which analyzis should be performed.
                                    (Default: AnalysisMode.ALL)
        anchor_mode (AnchorMode):   type of anchor generation. (Default:
                                    AnchorMode.PER_CLASS)
                                    ANCHOR mode only.
        anchor_count (int):         count of anchor per class (Default: 1).
                                    ANCHORS mode only
        percentiles (list):         list of tuples that should be used to calc.
                                    point cloud range and output shape.
                                    (Example: [(95,5), (96,4), ...]).
                                    PCD_RANGE mode only.
        voxel_sizes (list):         list of voxel sizes to use for evaluation.
                                    PCD_RANGE mode only.
    """

    def __init__(
            self,
            dataset: Det3DDataset,
            labels_path: str,
            points_path: str,
            modes: list = [AnalysisMode.ALL],
            anchor_mode: AnchorMode = AnchorMode.PER_CLASS,
            anchor_count: int = 1,
            percentiles: list = [(95, 5), (96, 4), (97, 3), (98, 2), (99, 1)],
            voxel_sizes: list = [[0.2, 0.2, 0.2], [0.3, 0.3, 4]]
    ):
        self.dataset = dataset
        self.labels_path = labels_path
        self.points_path = points_path
        self.anchor_mode = anchor_mode
        self.anchor_count = anchor_count
        self.modes = modes
        self.percentiles = percentiles
        self.voxel_sizes = voxel_sizes

        self.gt_frames = []
        self.gt_bboxes = []
        self.gt_labels = []
        self.gt_num_lidar_pts = []
        self.gt_truncated = []
        self.gt_occluded = []

        self.points_frames = []
        self.points_count = []
        self.points_range = []

    def analyze(self) -> None:
        """Wrapper to select correct anaylization mode by input"""
        if AnalysisMode.ALL in self.modes:
            self.generate_anchors()
            self.generate_point_cloud_range()
            self.analyze_class_usage()
        else:
            for mode in self.modes:
                if mode == AnalysisMode.ANCHORS:
                    self.generate_anchors()
                elif mode == AnalysisMode.PCD_RANGE:
                    self.generate_point_cloud_range()
                elif mode == AnalysisMode.CLASSES:
                    self.analyze_class_usage()
                else:
                    raise NotImplementedError(f"Mode {mode} is not implement!")

    def load_labels(self) -> None:
        """
            Loads labels from given path for evaluation.
            Use joblib cache for multiple runs.
        """
        print("Loading label files")

        @memory.cache
        def _load(labels_path):
            files = glob.glob(osp.join(labels_path, "*.json"))
            assert files, f"No JSON label files found at {labels_path}"

            gt_frames, gt_bboxes, gt_labels = [], [], []
            gt_num_lidar_pts, gt_truncated, gt_occluded = [], [], []

            for file in tqdm(files, unit="file"):
                frame_id = osp.splitext(osp.basename(file))[0]
                with open(file, "r") as f:
                    data = json.load(f)
                    for instance in data["instances"]:
                        gt_frames.append(frame_id)
                        gt_bboxes.append(instance["bbox_3d"])
                        gt_labels.append(instance["bbox_label_3d"])
                        gt_num_lidar_pts.append(instance["num_lidar_pts"])
                        gt_truncated.append(instance["truncated"])
                        gt_occluded.append(instance["occluded"])

            return (
                np.array(gt_frames),
                np.array(gt_bboxes, dtype=np.float32),
                np.array(gt_labels, dtype=np.int32),
                np.array(gt_num_lidar_pts, dtype=np.int32),
                np.array(gt_truncated),
                np.array(gt_occluded),
            )

        (self.gt_frames,
         self.gt_bboxes,
         self.gt_labels,
         self.gt_num_lidar_pts,
         self.gt_truncated,
         self.gt_occluded) = _load(self.labels_path)

        assert (len(self.gt_frames) ==
                len(self.gt_bboxes) ==
                len(self.gt_labels) ==
                len(self.gt_num_lidar_pts) ==
                len(self.gt_truncated) ==
                len(self.gt_occluded)
                ), "Failed to load labels. Length Missmatch"

        print(
            f"Loaded {len(np.unique(self.gt_frames))} frames "
            f"with {len(self.gt_labels)} instances in total."
        )

    def _process_pcd_file(self, file: str) -> dict:
        """
            Helper function to load and process single pcd file.
            Expected pcd dimension is x,y,z,intensity per row and binary format

            Args:
                file (str):     path to current pcd file

            Returns:
                dict            containing id, point count and range
        """
        frame_id = osp.splitext(osp.basename(file))[0]

        with open(file, "rb") as f:
            data = np.fromfile(f, dtype=np.float32)
            data = data.reshape(-1, 4)

            maxs = np.amax(data[:, :3], axis=0)
            mins = np.amin(data[:, :3], axis=0)

            pcd_range = {}
            full_range = mins.tolist() + maxs.tolist()
            pcd_range[(100, 0)] = full_range

            for percentile in self.percentiles:
                p_max, p_min = percentile

                mins = np.percentile(data[:, :3], p_min, axis=0)
                maxs = np.percentile(data[:, :3], p_max, axis=0)
                pcd_range[percentile] = mins.tolist() + maxs.tolist()
        return {
            "frame_id": frame_id,
            "count": data.shape[0],
            "range": pcd_range
        }

    def process_point_cloud(self) -> None:
        """
            Wrapper function to load and process all point cloud files.
            Use joblib cache for multiple runs
        """

        @memory.cache
        def _load(points_path):
            files = glob.glob(osp.join(points_path, "*.bin"))
            assert files, (
                f"No .bin point cloud files found at {self.points_path}"
            )

            frames = []
            points_count = []
            points_range = []
            with Pool(processes=24) as pool:
                for result in tqdm(
                        pool.imap_unordered(
                            self._process_pcd_file, files, chunksize=10),
                        total=len(files),
                        unit="file",
                        desc="Processing files"
                ):
                    frames.append(result["frame_id"])
                    points_count.append(result["count"])
                    points_range.append(result["range"])

                return (frames, points_count, points_range)

        (
            self.frames,
            self.points_count,
            self.points_range,
        ) = _load(self.points_path)

    def generate_anchors(self) -> None:
        """Wrapper for caling correct anchor generation function"""
        if not self.gt_labels:
            self.load_labels()

        if self.anchor_mode == AnchorMode.PER_DATASET:
            self.generate_anchors_per_dataset(self.anchor_count)
        elif self.anchor_mode == AnchorMode.PER_CLASS:
            self.generate_anchors_per_class(self.anchor_count)
        else:
            raise NotImplementedError(
                f"AnchorMode {self.anchor_mode} is not implemented")

    def generate_anchors_per_dataset(self, num_anchor: int) -> None:
        """
            Generates anchors based on labels using KMeans cluster using
            the whole dataset.

            Prints result to stdout.

            Args:
                num_anchor (int):       number of total anchor
        """
        self.anchor_head_sizes = {}

        bboxes_sizes = self.gt_bboxes[:, 3:6]
        bboxes_sizes = np.unique(bboxes_sizes, axis=0)

        print(
            f"Generate {num_anchor} anchor on {len(bboxes_sizes)} "
            "unique bbox sizes for whole dataset.."
        )
        # Kmeans Anchor
        if len(bboxes_sizes) < num_anchor:
            print(f"WARN: data has only "
                  f"{len(bboxes_sizes)} unique bbox sizes"
                  )
            anchors = bboxes_sizes
        else:
            kmeans = KMeans(n_clusters=num_anchor, n_init="auto",
                            random_state=0).fit(bboxes_sizes)
            anchors = kmeans.cluster_centers_

        anchors = self.sanitize_anchor_sizes(anchors)
        self.anchor_head_sizes["dataset"] = anchors
        iou_stats = {}
        iou_stats["dataset"] = self.test_anchor(anchors, self.gt_bboxes)

        self.print_anchor_head_sizes(self.anchor_head_sizes, iou_stats)

    def generate_anchors_per_class(
            self,
            anchor_per_class: int,
            include_testing: bool = False
    ) -> None:
        """
            Generates anchors based on labels using KMeans cluster per class.

            Prints result to stdout.

            Args:
                anchor_per_class (int):     number of anchor per class
                include_testing (bool):     test anchor coverage using IoU.
                                            Increases runtime.
                                            (Default: False)
        """
        self.anchor_head_sizes = {}
        iou_stats = {}

        print(f"Generate {anchor_per_class} anchor per class for "
              f"{len(self.dataset.META_INFO['classes'])} classes..")
        classes = list(enumerate(self.dataset.META_INFO["classes"]))
        for class_idx, class_name in tqdm(classes, unit="class", colour="green"):
            bboxes = self.gt_bboxes[self.gt_labels == class_idx]

            bboxes_sizes = bboxes[:, 3:6]
            bboxes_sizes = np.unique(bboxes_sizes, axis=0)

            # Kmeans Anchor
            if len(bboxes_sizes) < anchor_per_class:
                print(f"WARN: {class_name} has only "
                      f"{len(bboxes_sizes)} unique bbox sizes")
                anchors = bboxes_sizes
            else:
                kmeans = KMeans(n_clusters=anchor_per_class, n_init="auto",
                                random_state=0).fit(bboxes_sizes)
                anchors = kmeans.cluster_centers_

            self.anchor_head_sizes[class_name] = anchors

            # Test Anchor coverage
            if include_testing:
                tqdm.write(f"Testing Anchor for {class_name}...")
                iou_stats[class_name] = self.test_anchor(anchors, bboxes_sizes)

        self.print_anchor_head_sizes(self.anchor_head_sizes, iou_stats)

    def test_anchor(
            self,
            anchor_sizes: np.ndarray,
            bbox_sizes: np.ndarray,
    ) -> dict:
        """
        Tests coverage of given anchors on given anchors.
        Expects 3D bbox sizes with format [[dx,dy,dz]]

        Args:
            anchor_sizes (np.ndarray):  array containing all anchors
            bbox_sizes (np.ndarray):    containing all label bboxes

        Returns:
            dict with stats about the performance of the ious
        """
        ious = []
        for anchor_size in anchor_sizes:
            for bbox_size in tqdm(
                    bbox_sizes,
                    colour="red",
                    total=len(anchor_sizes) * len(bbox_sizes),
                    leave=False
            ):
                anchor_bbox = np.zeros((7,))
                anchor_bbox[3:6] = anchor_size

                bbox = np.zeros((7,))
                bbox[3:6] = bbox_size

                cuda_dev = torch.device("cuda:0")
                bbox_tensor = torch.Tensor(
                    bbox).to(cuda_dev).unsqueeze(0)
                anchor_tensor = torch.Tensor(
                    anchor_bbox).to(cuda_dev).unsqueeze(0)
                iou = diff_iou_rotated_3d(
                    bbox_tensor, anchor_tensor).detach().item()
                ious.append(iou)
        ious = np.array(ious)

        return {
            "uniques": len(anchor_sizes),
            "mean": np.mean(ious, axis=0),
            "std": np.std(ious, axis=0),
            "min": np.min(ious, axis=0),
            "max": np.max(ious, axis=0)
        }

    def print_anchor_head_sizes(
            self,
            anchor_head_sizes: dict,
            iou_stats: dict,
    ) -> None:
        """
            Helper function to pretty print all anchor head sizes.

            Args:
                anchor_head_sizes (dict):   dict per class name containing list
                                            of anchor sizes
                iou_stats (dict):           (optional) anchor performance value
        """
        print("Anchor Head Settings:")
        print("sizes=[")
        if not iou_stats:
            for (class_name, bbox_sizes) in anchor_head_sizes.items():
                print(f"\t# {class_name}")
                for bbox_size in bbox_sizes:
                    print(f"\t{bbox_size.tolist()},")
        else:
            data = zip(anchor_head_sizes.items(), iou_stats.items())
            for (class_name, bbox_sizes), (_, stats) in data:
                print(f"\t# {class_name}")
                print(
                    "\t# Coverage (IoU): "
                    f"{stats['mean']:.3f}|{stats['std']:.3f}|"
                    f"{stats['min']:.3f}|{stats['max']:.3f}|{stats['uniques']}"
                    "(mean|std|min|max|uniques)"
                )
                for bbox_size in bbox_sizes:
                    print(f"\t{bbox_size.tolist()},")
        print("],\n")

    def analyze_class_usage(self, low_point_thr: int = 15) -> None:
        """
            Analyzes the class balance in the annotations and the bbox
            statistics per class.

            Prints results to stdout.

            Args:
                low_point_thr (int):    threshold at which count as bbox with
                                        few points. (Default: 15)
        """
        if not self.gt_frames:
            self.load_labels()

        total_frames = len(np.unique(self.gt_frames))
        total_samples = len(self.gt_frames)
        print("Class Usage Per Class")
        classes_analysis = {}
        for class_id, class_name in tqdm(
                enumerate(self.dataset.METAINFO['classes'])
        ):
            filtered_idxs = self.gt_labels == class_id
            filtered_frames = self.gt_frames[filtered_idxs]
            filtered_points = self.gt_num_lidar_pts[filtered_idxs]

            num_total = len(filtered_frames)
            num_frames = len(np.unique(filtered_frames))
            freq = (
                num_total /
                total_samples
                if self.gt_frames.size
                else 0.0
            )

            num_low_points = np.sum(filtered_points <= low_point_thr)
            classes_analysis[class_name] = {
                "num_total": num_total,
                "num_frames": num_frames,
                "freq": freq,
                "min_points": (
                    filtered_points.min() if filtered_points.size else None
                ),
                "mean_points": (
                    filtered_points.mean() if filtered_points.size else None
                ),
                "max_points": (
                    filtered_points.max() if filtered_points.size else None
                ),
                "count_points": len(filtered_points),
                "low_points": num_low_points,
                "hist_points": np.histogram(filtered_points, bins=5),
            }

        # --- 1. Table Samples and Frames ---
        sample_table = []
        for cls, val in classes_analysis.items():
            sample_table.append([
                cls,
                val['num_total'],
                f"{val['freq']*100:.1f}%" if val['freq'] else "None",
                val['num_frames'],
                f"{val['num_frames']/total_frames*100:.1f}%",
            ])
        print("Sample Distribution:")
        print(tabulate(
            sample_table,
            headers=[
              "Class", "Samples", "Freq.", "Frames", "Frame Freq.",
              ],
            tablefmt="github",
        ))

        # --- 2. Table Num. Points per Bbox ---
        numpoints_table = []
        for cls, v in classes_analysis.items():
            numpoints_table.append([
                cls,
                (
                    f"{v['min_points']:.0f}"
                    if v['min_points'] is not None
                    else "None"
                ),
                (
                    f"{v['mean_points']:.0f}"
                    if v['mean_points'] is not None
                    else "None"
                ),
                (
                    f"{v['max_points']:.0f}"
                    if v['max_points'] is not None
                    else "None"
                ),
            ])
        print("\nPoint Statistics:")
        print(tabulate(numpoints_table, headers=[
              "Class", "Min Pts", "Mean Pts", "Max Pts"], tablefmt="github"))

        # --- 3. Table Low Points per BBox ---
        low_point_table = []
        for cls, v in classes_analysis.items():
            percent = (v['low_points'] / v['count_points']) * \
                100 if v['count_points'] else 0.0
            low_point_table.append([
                cls,
                v['low_points'],
                f"{percent:.2f}%"
            ])
        print(f"\nLow-Point BBoxes (≤ {low_point_thr} pts):")
        print(tabulate(low_point_table, headers=[
              "Class", "Low-Point Boxes", "Percent"], tablefmt="github"))

        # --- 4. Histogram ---
        print("\nHistograms Points per BBox")
        for class_name, values in classes_analysis.items():
            hist, bin_edges = values.get('hist_points')
            print(f"{class_name}:")
            max_count = max(hist)
            bar_char = "█"
            width = 30
            max_count = max(hist) if len(hist) else 1
            for count, left, right in zip(hist, bin_edges[:-1], bin_edges[1:]):
                bar_len = int((count / max_count) * width)
                bar = bar_char * bar_len
                percent = (count / values['count_points']) * \
                    100 if values['count_points'] else 0
                print(
                    f"  [{humanize(left):>6}, {humanize(right):>6}): "
                    f"{count:6} ({percent:5.2f}%) {bar}"
                )

            print()

    def generate_point_cloud_range(self) -> None:
        """
        Calculates point cloud statistis:
            - range (over all frames)
            - points per frame
            - SECOND output for different voxel sizes

        Prints result to stdout.
        """
        print("Point Cloud Range")
        if not self.points_frames:
            self.process_point_cloud()

        def text_hist(data: list, bins: int = 10, width: int = 40) -> None:
            """
            Print histogram of given data to stdout.

            Args:
                bins (int):     Number of bins to calculate. (Default: 10)
                width (int):    Text width of histogram. (Default: 40)
            """
            hist, bin_edges = np.histogram(data, bins=bins)
            max_count = max(hist)
            for i in range(len(hist)):
                bar = '#' * int(width * hist[i] / max_count)
                percent = hist[i] / max_count * 100
                print(
                    f"{int(bin_edges[i]):>7} - {int(bin_edges[i+1]):>6} "
                    f"{bar} "
                    f" ({percent:>5.1f}%)"
                )

        def calc_total_point_cloud_range(
                points_range: list,
                upper: int,
                lower: int,
        ) -> list:
            """
            Calculates total point cloud over all frames by percentile.

            Args:
                points_range (list):    List of dicts with range per frame
                                        and percentiles.
                upper (int):            upper percentile to aggregate
                lower (int):            lower percentile to aggreate

            Returns:
                list - point cloud range over all frame by given percentile
            """
            all_ranges = []
            for frame_dict in points_range:
                all_ranges.append(frame_dict[(upper, lower)])

            all_ranges = np.array(all_ranges)
            max_vals = np.amax(all_ranges[:, 3:], axis=0)
            min_vals = np.amin(all_ranges[:, :3], axis=0)

            total_range = np.round(
                np.concatenate([min_vals, max_vals]),
                2
            ).tolist()
            return total_range

        def round_up_to_multiple(value: int, divisor: int) -> int:
            """
            Rounds value to next int dividable number by given divisor.

            Example:
                value=9
                divisor=8
                returns 16
                16 // 8 = 0

            Args:
                value (int):    value to round up
                divisor (int):  divisor to test with

            Returns:
                rounded value (int)
            """
            return ((value + divisor - 1) // divisor) * divisor

        def calc_output_shape(
                point_cloud_range: list,
                voxel_size: list,
                downsample_strides: list = [1, 2, 4],
        ) -> list:
            """
            Calculates SECOND output shape to given voxel size and
            downsample strides.

            Args:
                point_cloud_range (list):   given point cloud range by format
                                   (x_min, y_min, z_min, x_max, y_max, z_max)
                voxel_size (list):          3D voxel size to test (x, y, z)
                downsample_strides (list)   used SECOND downsample strides.
                                            (Default: [1, 2, 4])
            Returns:
                output shape (list) in format [x, y]
            """
            x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
            vx, vy, vz = voxel_size

            total_downsample = 1
            for stride in downsample_strides:
                total_downsample *= stride

            x_size = math.ceil((x_max - x_min) / vx)
            y_size = math.ceil((y_max - y_min) / vy)

            x_size = round_up_to_multiple(x_size, total_downsample)
            y_size = round_up_to_multiple(y_size, total_downsample)

            return (x_size, y_size)  # (W, H) or (cols, rows)

        print("\nPoints per Frame:")
        text_hist(self.points_count, bins=10)

        print("\nPoint Cloud Ranges:")
        for upper, lower in [(100, 0)] + self.percentiles:
            pcd_range = calc_total_point_cloud_range(
                self.points_range, upper, lower)
            print(f"{(upper,lower)}:")
            print(pcd_range)
            for voxel_size in self.voxel_sizes:
                output_shape = calc_output_shape(pcd_range, voxel_size)
                print(f"Voxel size {voxel_size} - Output Shape {output_shape}")
            print()


class DatasetAnalyerArgumentParser(argparse.ArgumentParser):
    """
    CLI Argument Parser for DatasetAnalyzer uses argparse.ArgumentParser


    # Arguments
    name: str - Dataset name which is used in the default help texts
    labels_path: str - Path to parsed labels directory
    points_path: str - Path to parsed points directory

    Note: additional kwargs for ArgumentParser can be passed


    # Description
    The default description given to ArgumentParser is

    f"Analyze {name} dataset for openmmlabs mmdet3d usage"


    # Usage
    parser = DatasetAnalyerArgumentParser(
        "MyDataset",
        "data/mydataset/labels",
        "data/mydataset/points",
        "data/mydataset/",
    )
    args = parser.parse_args()
    """

    def __init__(
            self,
            name: str,
            labels_path: str,
            points_path: str,
            **kwargs,
    ):
        super().__init__(
            description=f"Analyze {name} dataset for openmmlabs mmdet3d usage",
            **kwargs
        )
        self.add_argument(
            '--labels',
            help=f"path of the converted {name} labels",
            default=labels_path
        )
        self.add_argument(
            '--points',
            help=f"path of the converted {name} points",
            default=points_path,
        )
        self.add_argument(
            "--anchor-mode",
            default=AnchorMode.PER_CLASS,
            type=lambda val: AnchorMode[val.upper()],
            choices=list(AnchorMode),
            help=(
                "choose 0 for anchor generation per class "
                "or 1 for anchor generation per dataset"
            )
        )
        self.add_argument(
            "--anchor-count",
            default="1",
            type=int,
            help=(
                "if anchor_mode is 0 it defines anchor per class. "
                "If anchor mode is 1 it defines total count of anchor"
            )
        )
        self.add_argument(
            "--mode",
            default=[AnalysisMode.ALL],
            type=lambda val: AnalysisMode[val.upper()],
            choices=list(AnalysisMode),
            help="choose modes to analyze dataset. Default all.",
            nargs="+",
        )
