import numpy as np
import glob
import json
import shutil
import pypcd4
from os import path as osp

from mmengine import print_log, dump
from mmengine.utils import track_parallel_progress
# Hardcoded to avoid mmcv._ext CUDA dependency
_EXP_DEFAULT_CLASSES = [
    "person", "car", "bicycle", "motorcycle", "bus", "truck", "unknown",
]


def convert_labels(root_path: str, out_dir: str, max_workers: int) -> None:
    """
    Converts openlabe format into openmmlab readable json format.

    Args:
        root_path (str): Path to dataset root.
        labels_path (str): Output directory of label json files.

    """
    print_log(f"Converting Labels to {out_dir}/labels")
    label_conv = ExpLabelConverter(
        root_path, out_dir, max_workers=max_workers)
    label_conv.convert()


def convert_pcd(root_path: str, out_dir: str, max_workers: int) -> None:
    """
    Converts PCD format into 4-D binary files.

    Included information are x,y,z and LiDAR intensity.

    Args:
        root_path (str): Path to dataset root.
        labels_path (str): Output directory of label json files.

    """
    print_log(f"Converting PCD to {out_dir}/points")
    pcd_conv = ExpPCDParser(root_path, out_dir, max_workers=max_workers)
    pcd_conv.convert()


def create_instance_dict(
        elem: str,
        data: dict,
        pcd_dimension: int = 4
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


def load_and_parse_labels(
        sample_list: list,
        root_path: str,
) -> list:
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
            file_path = osp.join(
                root_path, "labels", f"{elem}.json")
            with open(file_path, "r") as f:
                instance = create_instance_dict(elem, json.load(f))

                if not instance:
                    continue
                data_list.append(instance)
        except FileNotFoundError as e:
            print_log(f"WARN: Found no label to current frame {elem}: {e}")
            continue

    return data_list


def _split_by_bags(
    samples: list,
    val_tail_ratio: float = 0.2,
) -> tuple:
    """Split by fixed bag assignment with per-bag internal validation.

    Bag assignment (from page 14 of Zwischenvortrag):
        Bags 1,2,4,5,7,8  -> train (with last val_tail_ratio -> val)
        Bags 3,6,9         -> test  (all frames)

    Within each training bag, frames are sorted by frame_id and the
    last ``val_tail_ratio`` fraction is used for validation, while the
    first ``1 - val_tail_ratio`` fraction is used for training.

    The sample_id follows the pattern ``{bag}{sensor}0000{frame:06d}``
    where ``bag`` is the first character of sample_id.

    Args:
        samples (list):      list of sample dicts, each must have
                             a ``sample_id`` key
        val_tail_ratio (float): fraction of each train bag's tail
                                used for validation (Default: 0.2)

    Returns:
        tuple: (train_samples, val_samples, test_samples)
    """
    TRAIN_BAGS = {'1', '2', '4', '5', '7', '8'}
    TEST_BAGS = {'3', '6', '9'}

    # Group by bag (first character of sample_id)
    bag_samples: dict[str, list] = {}
    for s in samples:
        bag = s['sample_id'][0]
        bag_samples.setdefault(bag, []).append(s)

    # Sort each bag by frame_id (last 6 chars of sample_id)
    for bag in bag_samples:
        bag_samples[bag].sort(key=lambda s: int(s['sample_id'][-6:]))

    train, val, test = [], [], []
    for bag, items in bag_samples.items():
        if bag in TRAIN_BAGS:
            split_at = int(len(items) * (1.0 - val_tail_ratio))
            train.extend(items[:split_at])
            val.extend(items[split_at:])
        elif bag in TEST_BAGS:
            test.extend(items)
        else:
            print_log(
                f"WARN: unknown bag '{bag}' (sample_id={items[0]['sample_id']}), "
                f"assigning to test"
            )
            test.extend(items)

    return train, val, test


def create_train_val_test_split(
    data_path: str,
    out_dir: str,
    info_prefix: str,
    target_dataset: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> None:
    """Creates train/val/test splits by fixed bag assignment.

    Uses ``_split_by_bags`` to assign samples based on which
    experiment bag they originate from, ensuring no cross-bag
    leakage between train/val/test.

    Output files:
        {out_dir}/{info_prefix}_{target_dataset}_infos_{sensor}_{split}.pkl
        where sensor in {merged, os0, os1} and split in {train, val, test}

    Args:
        data_path (str):      root dir containing ``labels/`` subdir
        out_dir (str):        output directory for pkl files
        info_prefix (str):    prefix for annotation file names
        target_dataset (str): target dataset name (lumpi/kitti/osdar)
        train_ratio (float):  unused – kept for API compatibility
        val_ratio (float):    unused – kept for API compatibility
        seed (int):           unused – kept for API compatibility
    """
    for sensor, glob_pattern in [
        ("os0",    osp.join(data_path, "labels", "[0-9]0*.json")),
        ("os1",    osp.join(data_path, "labels", "[0-9]1*.json")),
        ("merged", osp.join(data_path, "labels", "[0-9]2*.json")),
    ]:
        label_files = glob.glob(glob_pattern)
        samples = load_and_parse_labels(
            [osp.basename(f).split(".")[0] for f in label_files],
            data_path,
        )

        train_samples, val_samples, test_samples = _split_by_bags(samples)

        for split_name, split_samples in [
            ("train", train_samples),
            ("val",   val_samples),
            ("test",  test_samples),
        ]:
            pkl_path = osp.join(
                out_dir,
                f"{info_prefix}_{target_dataset}_infos_{sensor}_{split_name}.pkl"
            )
            print_log(
                f"Dumping {len(split_samples)} samples to {pkl_path}"
            )
            dump(split_samples, pkl_path)


class ExpLabelConverter():
    """
    Wrapper class to convert Experiment labels into LUMPI dataset format

    Containing fixed label mapping:
        - "bike": "bicycle"
        - "person": "person"
        - "car": "car"


    Args:
        dataset_path (str):     path to raw dataset containing subdir dir of
                                each experiment with pcds and labels inside
        output_dir (str):       root dir to store parsed result (subdir labels)
        max_workers (int):      max. workers for parallization (Default: 8)
        chunk_size (int):       number of files per worker (Defult: 1000)
        classes (list):         metainfo element of classes
                                (Default: LUMPIDataset.METAINFO['classes'])
    """

    def __init__(
            self,
            dataset_path: str,
            output_dir: str,
            max_workers: int = 8,
            chunk_size: int = 1000,
            classes: dict = _EXP_DEFAULT_CLASSES,
    ):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.classes = classes

        self.class_mapping = {
            "bike": "bicycle",
            "person": "person",
            "car": "car",
        }

        self.measurements = glob.glob(
            f"{self.dataset_path}/*_experiment_*/")

    def convert(self) -> None:
        """Wrapper for calling convertion for each experiment"""

        for i, measurement in enumerate(self.measurements):
            print(f"[{i}/{len(self.measurements)}] Converting {measurement}..")
            self.process_measurement(measurement)

    def process_measurement(self, measurement: str) -> None:
        """
        Wrapper for calling convertiong for each configuration:
            - merged
            - sensor OS0
            - sensor OS1

        Prefers manually corrected labels from `*_labels_manual` directories
        over auto-generated `*_labels` directories. If a manual directory
        exists, its contents are used instead of the auto-generated ones.

        Args:
            measurement (str):  subdir name of current experiment
        """
        def get_label_dir(label_type: str) -> str:
            """Get the preferred label directory for a given label type.
            
            Prefers `*_labels_manual` over `*_labels`. Falls back to `*_labels`
            if no manual directory exists.
            
            Args:
                label_type (str): One of "merged", "os0", "os1"
                
            Returns:
                str: Path to the preferred label directory
            """
            manual_dir = osp.join(measurement, f"{label_type}_labels_manual")
            auto_dir = osp.join(measurement, f"{label_type}_labels")
            
            if osp.isdir(manual_dir):
                manual_files = glob.glob(osp.join(manual_dir, "*.json"))
                if manual_files:
                    print_log(
                        f"Using manually corrected labels: {manual_dir} "
                        f"({len(manual_files)} files)"
                    )
                    return manual_dir
            
            print_log(f"Using auto-generated labels: {auto_dir}")
            return auto_dir

        merged = glob.glob(osp.join(get_label_dir("merged"), "*.json"))
        os0 = glob.glob(osp.join(get_label_dir("os0"), "*.json"))
        os1 = glob.glob(osp.join(get_label_dir("os1"), "*.json"))

        track_parallel_progress(
            self.process_file, merged, nproc=self.max_workers
        )

        track_parallel_progress(
            self.process_file, os0, nproc=self.max_workers
        )
        track_parallel_progress(
            self.process_file, os1, nproc=self.max_workers
        )

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
        with open(file, "r") as f:
            data = json.load(f)

        return data

    def convert_to_instance(self, data: dict, file: str) -> dict:
        """
        Converting loaded annotation to mmdetection3d instance format

        Example of file naming:
            path: 4_experiment_car_1/os1_labels/81199.json
            orig filename: 81199.pcd
            filename: 4|1|0000|008119
                4       - Measurement
                1       - Type (0 - os0, 1 - os1, 2 - merged)
                0000    - Delimiter
                008119  - zero padded time in decaseconds

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
            frame_id = int(osp.basename(file).split(".")[0])
            if frame_id not in instances_by_frame:
                instances_by_frame[frame_id] = []

            required_keys = ['bbox', 'label', 'num_lidar_pts']
            if not all(key in row for key in required_keys):
                print(f"ERROR: {file} is missing a key! {required_keys}")
                continue

            if row['label'] not in self.classes:
                class_id = self.classes.index(self.class_mapping[row['label']])
            else:
                class_id = self.classes.index(row['label'])

            instances_by_frame[frame_id].append({
                "bbox_3d": row["bbox"],
                "bbox_label_3d": class_id,
                "num_lidar_pts": row["num_lidar_pts"],
                "occluded": 0.0,
                "truncated": 0.0,
            })

        return instances_by_frame

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
            type_name = osp.basename(osp.dirname(file))
            # Handle both *labels and *labels_manual directories
            if type_name.startswith("merged_labels"):
                type_number = 2
            elif type_name.startswith("os0_labels"):
                type_number = 0
            elif type_name.startswith("os1_labels"):
                type_number = 1
            else:
                raise ValueError(
                    f"Failed to extract type from {file}, "
                    f"type_name={type_name}. Expected directory name "
                    f"starting with 'merged_labels', 'os0_labels', "
                    f"or 'os1_labels'."
                )

            measurement_number = osp.basename(
                osp.dirname(osp.dirname(file)))[0]

            frame_id = int(osp.basename(file).split(".")[0])

            file_name = f"{measurement_number}{type_number}0000{frame_id:06d}"
            json_path = osp.join(self.output_dir, folder, f"{file_name}.json")
            with open(json_path, "w") as f:
                json.dump({"instances": instances}, f)


class ExpPCDParser:
    """
    Wrapper class to convert pointclouds from Experiment into MMDetection3D
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

        self.measurements = glob.glob(
            f"{self.dataset_path}/*_experiment_*/")

    def convert(self):
        """Wrapper for calling convertion for each experiment"""

        for i, measurement in enumerate(self.measurements):
            print(f"[{i}/{len(self.measurements)}] Converting {measurement}..")
            self.process_measurement(measurement)

    def process_measurement(
            self,
            measurement: str,
            chunksize: int = 100,
    ) -> None:
        """
        Wrapper for calling convertiong for each configuration:
            - merged
            - sensor OS0
            - sensor OS1

        Args:
            measurement (str):  subdir name of current experiment
            chunk_size (int):       number of files per worker (Defult: 100)
        """
        merged = glob.glob(osp.join(measurement, "merged_pcd", "*.pcd"))
        os0 = glob.glob(osp.join(measurement, "os0_pcd_transform", "*.pcd"))
        os1 = glob.glob(osp.join(measurement, "os1_pcd_transform", "*.pcd"))

        track_parallel_progress(
            self.process_file, merged,
            chunksize=chunksize, nproc=self.max_workers
        )
        track_parallel_progress(
            self.process_file, os0,
            chunksize=chunksize, nproc=self.max_workers
        )
        track_parallel_progress(
            self.process_file, os1,
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

    def load_pcd_file(self, file: str) -> pypcd4.PointCloud:
        """
        Wrapper to load pointcloud from file.

        Args:
            file (str):     path to point cloud file

        Returns:
            pypcd PointCloud
        """
        pcd_data = pypcd4.PointCloud.from_path(file)
        return pcd_data

    def convert_pcd(self, pcd_data: pypcd4.PointCloud, file: str) -> np.ndarray:
        """
        Converts point cloud into np.ndarray for binary storing.
        Uses dimension [x,y,z,intensity]

        Args:
            pcd_data (pypcd.PointCloud):        point cloud
            file (str):                         original file name

        Returns:
            np.ndarray with four dimensions
        """
        points = np.zeros([len(pcd_data.pc_data), 4], dtype=np.float32)

        x = pcd_data.pc_data['x'].copy()
        y = pcd_data.pc_data['y'].copy()
        z = pcd_data.pc_data['z'].copy()

        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = z
        points[:, 3] = pcd_data.pc_data['intensity'].copy().astype(np.float32)

        return points

    def store_to_file(
        self,
            points: np.ndarray,
            file: str,
            folder: str = "points",
    ) -> None:
        """
        Stors point cloud as binary file.

        Example:
            path: 4_experiment_car_1/os1_pcd/81199.pcd
            orig filename: 81199.pcd
            filename: 4|1|0000|008119
                4       - Measurement
                1       - Type (0 - os0, 1 - os1, 2 - merged)
                0000    - Delimiter
                008119  - zero padded time in decaseconds

        Args:
            points (np.ndarray):    4d array with point cloud data
            file (str):             original file name
            folder (str):           subdir to store result (Default: points)
        """
        type_name = osp.basename(osp.dirname(file))
        if type_name == "merged_pcd":
            type_number = 2
        else:
            assert type_name.startswith("os"), (
                f"Failed to extract new file name for {file}, {type_name}"
            )
            type_number = int(type_name[2])

        measurement_number = osp.basename(osp.dirname(osp.dirname(file)))[0]

        frame_id = int(osp.basename(file).split(".")[0])

        file_name = f"{measurement_number}{type_number}0000{frame_id:06d}"
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
