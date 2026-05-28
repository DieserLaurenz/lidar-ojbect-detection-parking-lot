import numpy as np
import glob
import json
import csv
import shutil
import random
import re
from os import path as osp
from plyfile import PlyData
from collections import defaultdict

from mmengine import print_log, dump
from mmengine.utils import track_parallel_progress
from mmdet3d.datasets import lumpi


def convert_labels(root_path: str, out_dir: str, max_workers: int) -> None:
    """Converts openlabe format into openmmlab readable json format.

    Args:
        - root_path (str): Path to dataset root.
        - labels_path (str): Output directory of label json files.

    """
    print_log(f"Converting Labels to {out_dir}/labels")
    label_conv = LumpiLabelConverter(
        root_path, out_dir, max_workers=max_workers)
    label_conv.convert()


def convert_pcd(root_path: str, out_dir: str, max_workers: int) -> None:
    """Converts PCD format into 4-D binary files.

    Included information are x,y,z and LiDAR intensity.

    Args:
        - root_path (str): Path to dataset root.
        - labels_path (str): Output directory of label json files.

    """
    print_log(f"Converting PCD to {out_dir}/points")
    pcd_conv = LumpiPCDParser(root_path, out_dir, max_workers=max_workers)
    pcd_conv.convert()


def create_instance_dict(
        elem: str,
        data: dict,
        pcd_dimension: int = 4,
) -> dict:
    """
    Create dict for single instance.

    Note:
        - sample_idx will be overwritten and is the enurmation
        - sample_id is introducted to store the frame id
        - num_pts_feats means point cloud dimension

    Args:
        elem (str):             current frame id
        data (data):            dict containing all annotations for this
                                frame
        pcd_dimension (int):    dimension of pcd files.
                                (Default: 4 for [x,y,z,intensitz])

    Returns:
        dict ready formatted
    """
    if not data["instances"]:
        print_log(f"WARNING: no instances found for {elem}")
        return {}

    instances = {
        "sample_id": elem,
        "lidar_points": {
            "num_pts_feats": pcd_dimension,
            "lidar_path": f"{elem}.bin"
        },
        "instances": data["instances"],
    }

    return instances


def load_and_parse_labels(sample_list: list, root_path: str) -> list:
    """
    Load and parse converted annotations. (Expected format: json)

    Args:
        sample_list (list):     list of frame ids
        root_path (str):        root path of dataset (subdir will be labels)

    Returns:
        List of contents of annotation files
    """
    data_list = []
    for elem in sample_list:
        try:
            file_path = osp.join(root_path, "labels", f"{elem}.json")
            with open(file_path, "r") as f:
                instance = create_instance_dict(elem, json.load(f))

                if not instance:
                    continue
                data_list.append(instance)
        except FileNotFoundError as e:
            print_log(f"WARN: Found no label to current frame {elem}: {e}")
            continue

    return data_list


def split_over_frames(
        all_samples: list,
        test_ratio: float,
        val_ratio: float,
) -> (list, list, list):
    """
    Split training, validation and test data by frames ignoring measurements.

    Args:
        all_samples (list):     list of all frame ids
        test_ratio (float):     ratio how much sample for testing
        val_ratio (float):     ratio how much sample for validation

    Returns:
        tuple with lists frame ids for
            training,
            validation,
            testing
    """
    num_total = len(all_samples)

    # Split 1: Training/Test
    num_test = int(num_total * test_ratio)
    num_trainval = num_total - num_test

    # Split 2: Validation from TrainVal
    num_val = int(num_trainval * val_ratio)
    num_train = num_trainval - num_val

    random.shuffle(all_samples)

    train_samples = all_samples[:num_train]
    val_samples = all_samples[num_train:num_train + num_val]
    test_samples = all_samples[-num_test:]

    return train_samples, val_samples, test_samples


def split_over_scenes(
        all_samples: list,
        test_ratio: float,
        val_ratio: float,
) -> (list, list, list):
    """
    Split training, validation and test data by measurements.
    Note: If measurements are uneven in length or class distribution
    it will impact the split!

    Args:
        all_samples (list):     list of all frame ids
        test_ratio (float):     ratio how much sample for testing
        val_ratio (float):     ratio how much sample for validation

    Returns:
        tuple with lists frame ids for
            training,
            validation,
            testing
    """
    scene_frames = defaultdict(list)
    naming_pattern = re.compile(r"([0-9])+0000([0-9]{6})")
    for sample in all_samples:
        naming_matches = naming_pattern.search(sample)
        if naming_matches:
            scene_id, frame_id = naming_matches.groups()
            scene_frames[scene_id].append(sample)

    scene_ids = list(scene_frames.keys())
    num_total = len(scene_ids)

    # Split 1: Training/Test
    num_test = int(num_total * test_ratio)
    num_trainval = num_total - num_test

    # Split 2: Validation from TrainVal
    num_val = int(num_trainval * val_ratio)
    num_train = num_trainval - num_val

    random.shuffle(scene_ids)

    train_samples = sum([
        scene_frames[scene_id]
        for scene_id in scene_ids[:num_train]
    ], [])
    val_samples = sum([
        scene_frames[scene_id]
        for scene_id in scene_ids[:num_train:num_train + num_val]
    ], [])
    test_samples = sum([
        scene_frames[scene_id]
        for scene_id in scene_ids[-num_test:]
    ], [])

    return train_samples, val_samples, test_samples


def load_splitset_from_file(
        splitset_train_path: str,
        splitset_val_path: str,
        splitset_test_path: str,
        data_path: str
) -> (list, list, list):
    """
    Load given splitset from files.
    Expects one frame id per line.

    Args:
        splitset_train_path (str):  path to file for set of training
        splitset_val_path (str):    path to file for set of validation
        splitset_test_path (str):   path to file for set of testing
        data_path (str):            root path of dataset
    """
    def read_splitset_file(path: str) -> list:
        """
        Reading splitset from file. One frame id per line.

        Args:
            path (str):     path to file

        Returns:
            list of frame ids
        """
        with open(path, "r") as f:
            return [int(line.strip()) for line in f if line.strip()]

    def get_label_files_name_by_scene(data_path: str, scene_id: str) -> list:
        """
        Reads all frame ids by given experiment id.

        Args:
            data_path (str):    root path of dataset
            scene_id (str):     experiment id

        Returns:
            list of all frame ids in this experiment
        """
        label_glob = glob.glob(f"{data_path}/labels/{scene_id}0000*.json")
        return [osp.splitext(osp.basename(file))[0] for file in label_glob]

    train_split = read_splitset_file(splitset_train_path)
    val_split = read_splitset_file(splitset_val_path)
    test_split = read_splitset_file(splitset_test_path)

    train_samples = get_label_files_name_by_scene(data_path, train_split)
    val_samples = get_label_files_name_by_scene(data_path, val_split)
    test_samples = get_label_files_name_by_scene(data_path, test_split)

    return train_samples, val_samples, test_samples


def create_train_val_test_split(data_path: str,
                                info_train_path: str,
                                info_val_path: str,
                                info_test_path: str,
                                test_ratio: float = 0.1,
                                val_ratio: float = 0.02,
                                limit_measurements_list: list = [],
                                split_type: str = "scene",
                                ) -> None:
    """
    Split dataset into training, validation and test subsets.

    Note:
        - train_ratio: 1 - test_ratio
        - whole dataset split into train and test samples by ratios
        - val samples are part of train samples

    Args:
        labels_path (str):      Path to converted labels.
        info_train_path (str):  Output path for annotation infos for training.
        info_val_path (str):    Output path for annotation infos for validation
        info_test_path (str):   Output path for annotation infos for testing.
        test_ratio (float):     percentale size of test samples.
                                (0<=x<=1) Default: 0.3
        val_ratio (float):      percentale size of val samples.
                                (0<=x<=1) Default: 0.2
        split_type(str):        decides whether to split the
                                data based on frames or scenes.
                                (Valid are frame or scene)
    """

    limit_measurements_set = set(limit_measurements_list)
    label_glob = glob.glob(f"{data_path}/labels/*.json")
    if limit_measurements_set:
        label_files = []
        for file in label_glob:
            measurement_id = osp.basename(file)[0]

            if measurement_id in limit_measurements_set:
                label_files.append(file)
    else:
        label_files = label_glob

    all_samples = [osp.splitext(osp.basename(file))[0]
                   for file in label_files]

    splitset_train_path = f"{data_path}/splitsets/train.txt"
    splitset_val_path = f"{data_path}/splitsets/val.txt"
    splitset_test_path = f"{data_path}/splitsets/test.txt"

    if (
        osp.isfile(splitset_train_path) and
        osp.isfile(splitset_val_path) and
        osp.isfile(splitset_test_path)
    ):
        print("Loading split set from dataset directory...")
        (
            train_sample_ids,
            val_sample_ids,
            test_sample_ids
        ) = load_splitset_from_file(
            splitset_train_path,
            splitset_val_path,
            splitset_test_path,
            data_path
        )

    else:
        print(f"Generating split set by {split_type}...")
        if split_type == "frame":
            (
                train_sample_ids,
                val_sample_ids,
                test_sample_ids
            ) = split_over_frames(all_samples)
        elif split_type == "scene":
            (
                train_sample_ids,
                val_sample_ids,
                test_sample_ids
            ) = split_over_scenes(all_samples)
        else:
            raise ValueError(f"Unknown split type {split_type}")

    # Assign splits
    train_samples = load_and_parse_labels(train_sample_ids, data_path)
    val_samples = load_and_parse_labels(val_sample_ids, data_path)
    test_samples = load_and_parse_labels(test_sample_ids, data_path)

    # Dump to files
    dump(train_samples, info_train_path)
    dump(val_samples, info_val_path)
    dump(test_samples, info_test_path)


class LumpiLabelConverter:
    """
    Wrapper class to convert LUMPI labels into MMDetection3D format

    Args:
        dataset_path (str):     path to raw dataset containing subdir dir of
                                each experiment with pcds and labels inside
        output_dir (str):       root dir to store parsed result (subdir labels)
        max_workers (int):      max. workers for parallization (Default: 8)
        chunk_size (int):       number of files per worker (Defult: 1000)
    """

    def __init__(
            self,
            dataset_path: str,
            output_dir: str,
            max_workers: int = 8,
            chunk_size: int = 1000,
    ):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.chunk_size = chunk_size

        labels_path = osp.join(
            self.dataset_path, "Label", "Measurement*", "Label.csv"
        )
        self.files = glob.glob(labels_path)
        assert self.files, f"No labels found at {labels_path}"

    def convert(self) -> None:
        """Wrapper for calling parallel convertion"""

        track_parallel_progress(
            self.process_file, self.files, nproc=self.max_workers)

    def process_file(self, file: str) -> None:
        """
        Wrapper for loading, converting and storing single file.

        Args:
            file (str):     path to file
        """
        data = self.load_label_file(file)
        instances = self.convert_to_instance(data, file)
        self.store_to_file(instances, file)

    def load_label_file(self, file: str) -> dict:
        """
        load content of label file. expects json.

        args:
            file (str):     path to file

        returns:
            dict of label content
        """
        data = []
        field_names = ["time", "object id",
                       "2d_rectangle_top_left_x",
                       "2d_rectangle_top_left_y",
                       "2d_rectangle_width",
                       "2d_rectangle_height",
                       "score", "class_id", "visibility",
                       "3d_box_center_x",
                       "3d_box_center_y",
                       "3d_box_center_z",
                       "3d_box_length",
                       "3d_box_width",
                       "3d_box_height",
                       "3d_box_heading",
                       "additionals",
                       ]
        with open(file, "r") as f:
            rows = csv.DictReader(f, fieldnames=field_names)
            next(rows)  # skip header row
            data = [data for data in rows]

        return data

    def convert_to_instance(self, data: dict, file: str) -> dict:
        """
        Converting loaded annotation to mmdetection3d instance format

        Example of file naming:
            timestamp: 811.900
            filename: 4 | 0000 | 008119
                4 - Measurement
                0000 - Delimiter
                008119 - zero padded time in decaseconds

        Required_keys in dict are ['bbox', 'label', 'num_lidar_pts'].

        Sets to occlusion and truncation to 0 since LUMPI does not
        provide information.

        Args:
            data (dict):    loaded annotation
            file (str):     original filename

        Returns:
            dict with converted annotation
        """
        instances_by_frame = {}
        for row in data:
            frame_id = int(float(row["time"]) * 10)
            if frame_id not in instances_by_frame:
                instances_by_frame[frame_id] = []

            instances_by_frame[frame_id].append({
                "bbox_3d": self.format_bbox(row),
                "bbox_label_3d": int(row["class_id"]),
                "num_lidar_pts": int(float(row["visibility"])),
                "occluded": 0.0,
                "truncated": 0.0,
            })

        return instances_by_frame

    def format_bbox(self, row: dict) -> list:
        """
        Convert dict of annotation into 7 element bbox
        with [x,y,z,dx,dy,dz,yaw]

        Args:
            row (dict):     content of single annotation

        Returns:
            7 element bbox notation
        """
        return [float(row["3d_box_center_x"]),
                float(row["3d_box_center_y"]),
                float(row["3d_box_center_z"]),
                float(row["3d_box_length"]),
                float(row["3d_box_width"]),
                float(row["3d_box_height"]),
                float(row["3d_box_heading"]),
                ]

    def store_to_file(
        self,
        instances_by_frame: dict,
        file: str,
        folder="labels",
    ) -> None:
        """
        Storing converted annotation to file.

        Args:
            instnaces_by_frame (dict):  converted instance indexed by frame id
            file (str):                 original file path
            folder (str):               subfolder name (Default: labels)
        """
        for frame_id, instances in instances_by_frame.items():
            measurement_number = osp.basename(osp.dirname(file))[-1]
            label_file_path = osp.join(
                self.output_dir,
                folder,
                f"{measurement_number}0000{frame_id:06d}.json"
            )
            with open(label_file_path, "w") as f:
                json.dump({"instances": instances}, f)


class LumpiPCDParser():
    """
    Wrapper class to convert pointclouds from LUMPI into MMDetection3D
    binary format.

    Args:
        dataset_path (str):     path to raw data. Expecting subdir for each
                                experiment.
        output_dir (str):       root dir to store results (subdir points)
        max_workers (int):      max. workers for parallization (Default: 8)
    """

    def __init__(
        self,
        dataset_path: str,
        output_dir: str,
        max_workers: int = 8
    ):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_workers = max_workers

        self.classes = lumpi.META_INFO["classes"]
        self.measurements = glob.glob(
            f"{self.dataset_path}/Points/Measurement*/")

    def convert(self):
        """Wrapper for calling convertion for each measurement"""

        for i, measurement in enumerate(self.measurements):
            print(f"[{i}/{len(self.measurements)}] Converting {measurement}..")
            self.process_measurement(measurement)

    def process_measurement(
            self,
            measurement: str,
            chunksize: int = 100,
    ) -> None:
        """
        Wrapper for calling convertiong for all point cloud files.

        Args:
            measurement (str):  subdir name of current experiment
            chunk_size (int):       number of files per worker (Defult: 100)
        """
        files = glob.glob(osp.join(measurement, "lidar/*.ply"))
        track_parallel_progress(
            self.process_file, files,
            chunksize=chunksize, nproc=self.max_workers
        )

    def process_file(self, file: str) -> None:
        """
        Wrapper for loading, converting and storing single file.

        Args:
            file (str):     path to file
        """
        data = self.load_pcd_file(file)
        points = self.convert_pcd(data, file)
        self.store_to_file(points, file)

    def load_pcd_file(self, file: str) -> dict:
        """
        Wrapper to load pointcloud from file.

        Args:
            file (str):     path to point cloud file

        Returns:
            dict with keys x,y,z,intensity
        """
        with open(file, 'rb') as f:
            plydata = PlyData.read(f)
            vertex = plydata['vertex']
            data = {
                name: vertex.data[name].copy()
                for name in ['x', 'y', 'z', 'intensity']
            }
        return data

    def convert_pcd(self, pcd_data: dict, file: str) -> np.ndarray:
        """
        Converts point cloud into np.ndarray for binary storing.
        Uses dimension [x,y,z,intensity]

        Args:
            pcd_data (dict):        point cloud
            file (str):             original file name

        Returns:
            np.ndarray with four dimensions
        """
        points = np.column_stack((
            pcd_data['x'], pcd_data['y'], pcd_data['z'], pcd_data['intensity']
        )).astype(np.float32)
        return points

    def store_to_file(
        self,
            points: np.ndarray,
            file: str,
            folder: str = "points",
    ) -> None:
        """
        Stors point cloud as binary file.

        Example of file naming:
            orig filename: 81199.py
            filename: 4 | 0000 | 008119
                4 - Measurement
                0000 - Delimiter
                008119 - zero padded time in decaseconds

        Args:
            points (np.ndarray):    4d array with point cloud data
            file (str):             original file name
            folder (str):           subdir to store result (Default: points)
        """
        frame_id = int(osp.splitext(osp.basename(file))[0])
        measurement_number = osp.basename(osp.dirname(osp.dirname(file)))[-1]
        file_name = f"{measurement_number}0000{frame_id:06d}"
        pcd_path = osp.join(self.output_dir, folder, f"{file_name}.bin")

        # Write to RAM and move if enough available
        free_mb = shutil.disk_usage("/dev/shm").free / 1024 / 1024
        if free_mb > 500:
            pcd_path_tmp = osp.join("/dev/shm", file_name)

            with open(pcd_path_tmp, "wb", buffering=1024*1024) as f:
                f.write(points.tobytes())

            shutil.move(pcd_path_tmp, pcd_path)
        else:
            with open(pcd_path, "wb", buffering=1024*1024) as f:
                f.write(points.tobytes())
