import random
import glob
import json
import math
import numpy as np
from collections import defaultdict
from pypcd4 import PointCloud as _PCD
from concurrent import futures
from os import path as osp
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

import mmengine
from mmengine import print_log
from mmengine.utils import track_parallel_progress
from mmdet3d.datasets import osdar23


def convert_labels(root_path: str, out_dir: str, max_workers: int) -> None:
    """Converts openlabe format into openmmlab readable json format.

    Args:
        - root_path (str): Path to dataset root.
        - labels_path (str): Output directory of label json files.

    """
    print_log(f"Converting Labels to {out_dir}/labels")
    label_conv = OSDaR23LabelConverter(
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
    pcd_conv = OSDaR23PCDConverter(root_path, out_dir, max_workers=max_workers)
    pcd_conv.convert()


def create_train_val_test_split(data_path: str,
                                info_train_path: str,
                                info_val_path: str,
                                info_test_path: str,
                                test_ratio: float = 0.1,
                                val_ratio: float = 0.1,
                                mode: str = "measurement",
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
                                (Valid are frame or measurement)
    """

    assert mode in ["measurement", "frame"], \
        "Mode needs to be measurement or frame!"

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
            "axis_align_matrix": data["axis_align_matrix"],

        }

        return instances

    def load_and_parse_labels(sample_list: list, root_path: str) -> list:
        """
        Load and parse converted annotations. (Expected format: json)

        Args:
            sample_list (list):     list of frame ids
            root_path (str):        root path of dataset, subdir will be labels

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

    if mode == "frame":
        label_files = glob.glob(f"{data_path}/labels/*.json")

        all_samples = [osp.splitext(osp.basename(file))[0]
                       for file in label_files]

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
    elif mode == "measurement":
        label_files = glob.glob(f"{data_path}/labels/*.json")
        measurements = defaultdict(list)
        for file in label_files:
            measurement_id = osp.basename(file).split("0000")[0]
            measurements[measurement_id].append(
                osp.splitext(osp.basename(file))[0])
        print(f"Found {len(measurements)} Measurements..")

        measurement_ids = list(measurements.keys())
        num_total = len(measurement_ids)

        # Split 1: Training/Test
        num_test = int(num_total * test_ratio)
        num_trainval = num_total - num_test

        # Split 2: Validation from TrainVal
        num_val = int(num_trainval * val_ratio)
        num_train = num_trainval - num_val

        random.shuffle(measurements)
        train_samples = sum([
            measurements[id]
            for id in measurement_ids[:num_train]
        ],
            [],
        )
        val_samples = sum([
            measurements[id]
            for id in measurement_ids[num_train:num_train + num_val]
        ],
            [],
        )
        test_samples = sum([
            measurements[id]
            for id in measurement_ids[-num_test:]
        ],
            [],
        )

        assert num_total == (num_train+num_val+num_test)
        assert len(label_files) == (
            len(train_samples) +
            len(val_samples) +
            len(test_samples)
        )

    else:
        raise NotImplementedError(f"Split Mode '{mode}' not implemeneted")

    train_samples = load_and_parse_labels(train_samples, data_path)
    mmengine.dump(train_samples, info_train_path)
    val_samples = load_and_parse_labels(val_samples, data_path)
    mmengine.dump(val_samples, info_val_path)
    test_samples = load_and_parse_labels(test_samples, data_path)
    mmengine.dump(test_samples, info_test_path)


class OSDaR23LabelConverter:
    """
    Wrapper class to convert OSDaR23 labels into MMDetection3D format

    Args:
        dataset_path (str):     path to raw dataset containing subdir dir of
                                each experiment with pcds and labels inside
        output_dir (str):       root dir to store parsed result (subdir labels)
        max_workers (int):      max. workers for parallization (Default: 8)
        dry_run (bool):         run without storing results
    """

    def __init__(
            self,
            root_path: str,
            out_dir: str,
            max_workers: int = 8,
            dry_run: bool = False,
    ):
        self.out_dir = out_dir
        self.files = glob.glob(f"{root_path}/*/*_labels.json")
        self.max_workers = max_workers
        self.dry_run = dry_run

    def convert(self):
        """Wrapper for calling parallel convertion"""

        track_parallel_progress(
            self.process_file, self.files, nproc=self.max_workers
        )

    def compute_axis_aligned_matrix(self, data: dict) -> np.ndarray:
        """
        Computes axis aligned matrix by using translation and quaterion.

        Args:
            data (dict):        label data

        Returns:
            4x4 matrix
        """
        translation = data["pose_wrt_parent"]["translation"]
        quaternion = data["pose_wrt_parent"]["quaternion"]

        T_align = np.eye(4)
        T_align[:3, :3] = R.from_quat(quaternion).as_matrix()
        T_align[:3, 3] = translation

        return T_align

    def quaternion_to_yaw(
            self,
            qx: float,
            qy: float,
            qz: float,
            qw: float,
    ) -> float:
        """
        Convert quaternion to yaw (rotation around z-axis)
        Yaw is derived from the quaternion

        Args:
            qx (float):     x component of quaternion
            qy (float):     y component of quaternion
            qz (float):     z component of quaternion
            qw (float):     w component of quaternion

        Returns:
            float yaw value
        """
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return yaw

    def convert_10_to_7_format(self, obj: list) -> list:
        """
        Converts 10 element bbox quaternion format into
        7 element bbox format with [x,y,z,dx,dy,dz,yaw]

        Args:
            obj (list):     bbox notation with quaternion

        Returns:
            7 element bbox format
        """
        x, y, z, qx, qy, qz, qw, sx, sy, sz = obj
        yaw = self.quaternion_to_yaw(qx, qy, qz, qw)

        return [x, y, z, sx, sy, sz, yaw]

    def extract_lidar_objects(self, objects: dict) -> list:
        """
        Extract annotation that contain cuboid labels.

        Args:
            objects (dict):     annotation of current frame

        Returns:
            list of all frame ids with lidar information
        """
        lidar_objects = []

        for elem_id, elem in objects.items():
            for data_id in elem["object_data_pointers"]:
                data_type = elem["object_data_pointers"][data_id]["type"]
                if (
                    data_id.startswith("lidar__cuboid") and
                    data_type == "cuboid"
                ):
                    lidar_objects.append(elem_id)

        return lidar_objects

    def convert_attributes(self, object_data: dict) -> dict:
        """
        Convert Attributes into mmdetection format occlusion and truncated.

        occlusion_mapping = {
            "0-25 %": 0.25,
            "25-50 %": 0.5,
            "50-75 %": 0.75,
            "75-99 %": 0.99,
            "100 %": 1.0,
        }

        Notation for truncated:
            0 = fully visible
            1 = partly occluded
            2 = largely occluded
            3 = unknown

        Args:
            object_data (dict):     dict of attributes for current element

        Returns:
            dict of values occlusion and trucated
        """
        result = {"occlusion": 0.0, "truncated": 0}
        if "attributes" not in object_data:
            print_log("WARN: No attributes in object data!")
            return

        occlusion_mapping = {
            "0-25 %": 0.25,
            "25-50 %": 0.5,
            "50-75 %": 0.75,
            "75-99 %": 0.99,
            "100 %": 1.0,
        }

        for item in object_data["attributes"].get("text", []):
            name, val = item.values()
            if name == "occlusion":
                result["occlusion"] = occlusion_mapping[val]
                if val not in occlusion_mapping:
                    print_log(f"WARN: unknown occlusion value: {val}!")

        for item in object_data["attributes"].get("boolean", []):
            if item.get("name") in ["isTrunactedBottom", "isTruncatedTop"]:
                result["truncated"] += 1

        return result

    def is_point_in_bbox(self, points: np.ndarray, bbox: list) -> np.ndarray:
        """
        Check if a point is inside a 3D bounding box.

        Args:
            points (np.ndarray):    array of points to test (3D)
            bbox (list):            bbox to test in 7 element format

        Returns:
            np.ndarray for each elem True if point is inside else False
        """
        bbox_center = bbox[:3]
        bbox_dims = bbox[3:6]
        bbox_yaw = bbox[-1]

        rot_matrix = R.from_euler('z', -bbox_yaw).as_matrix()

        translated_points = points - bbox_center

        local_points = translated_points @ rot_matrix.T

        w, l, h = bbox_dims
        return (
            (-w / 2 <= local_points[:, 0]) & (local_points[:, 0] <= w / 2) &
            (-l / 2 <= local_points[:, 1]) & (local_points[:, 1] <= l / 2) &
            (-h / 2 <= local_points[:, 2]) & (local_points[:, 2] <= h / 2)
        )

    def count_points_in_bbox(self, file: str, frame: str, bbox: list) -> int:
        """
        Counts point inside a given bbox using KDTree.

        Args:
            file (str):     path to file
            frame (str):    frame id
            bbox (list):    bbox to test in 7 element format

        Returns:
            int number of points inside bbox
        """
        pcd_file = self.generate_file_path(
            frame, file, folder="points", suffix="bin")

        point_cloud = np.fromfile(
            pcd_file, dtype=np.float32).reshape(-1, 4)[:, :3]
        tree = KDTree(point_cloud)

        radius = np.linalg.norm(bbox[3:6])

        indicies = tree.query_ball_point(bbox[:3], radius)
        points_in_bbox = point_cloud[indicies]

        counter = np.sum(self.is_point_in_bbox(points_in_bbox, bbox))
        return int(counter)

    def convert_objects_in_frame(
            self,
            lidar_objects: list,
            frame: str,
            objects: dict,
            file: str,
    ) -> list:
        """
        Converts objects in frame into MMDetection3D instance format.

        Args:
            lidar_objects (list):   list of frame ids with lidar objects
            frame (str):            frame id of current frame
            objects (dict):         dict of all objects
            file (str):             original file name

        Returns:
            list of all instances
        """
        instances = []
        for lidar_object_id in lidar_objects:
            # skip if object is not in current frame
            if lidar_object_id not in frame["objects"]:
                continue

            object_data = frame["objects"][lidar_object_id]["object_data"]

            # skip if no lidar information in current frame!
            if "cuboid" not in object_data:
                continue
            if len(object_data["cuboid"]) > 1:
                print_log("WARN: more than 1 vec object!")

            class_name = objects[lidar_object_id]["type"]
            # skip if class is excluded!
            if class_name not in osdar23.META_INFO['classes']:
                continue
            class_id = osdar23.META_INFO['classes'].index(class_name)

            bbox_3d = self.convert_10_to_7_format(
                object_data["cuboid"][0]["val"])
            attributes = self.convert_attributes(object_data["cuboid"][0])
            num_lidar_pts = self.count_points_in_bbox(file, frame, bbox_3d)

            instances.append({
                "bbox_3d": bbox_3d,
                "bbox_label_3d": class_id,
                "num_lidar_pts": num_lidar_pts,
                "truncated": attributes["truncated"],
                "occluded": attributes["occlusion"],
            })

        return instances

    def generate_file_path(
            self,
            frame,
            file,
            folder="labels",
            suffix="json",
    ) -> str:
        """
        Helper function to generate file path to label or point cloud
        by given frame id.

        Args:
            frame (str):    current frame id
            file (str):     full path to original file
            folder (str):   subdir to search labels or points
            suffix (str):   file suffix json or bin

        Retruns:
            str of file path
        """
        lidar_file = (
            frame["frame_properties"]["streams"]["lidar"]["uri"]
        ).split("/")[-1][:3]
        scene_number = (
            file.replace("_labels.json", "")
        ).split("_")[-1].replace(".", "")
        file_path = "%s/%s/%s0000%s.%s" % (
            self.out_dir, folder, scene_number, lidar_file, suffix
        )

        return file_path

    def load_label_file(self, file: str) -> dict:
        """
        load content of label file. expects json.

        args:
            file (str):     path to file

        returns:
            dict of label content
        """
        data = []
        with open(file, "r") as f:
            data = json.load(f)["openlabel"]

        return data

    def store_to_file(
        self,
        instances: list,
        frame: str,
        file: str,
        axis_align_matrix: np.ndarray,
    ) -> None:
        """
        Storing converted annotation to file.

        Args:
            instances (list):       converted list of instances
            frame (str):            current frame id
            file (str):             original file path
            axis_align_matrix:      4x4 axis align matrix
        """
        frame_label_file = self.generate_file_path(frame, file)

        with open(frame_label_file, "w") as f:
            json.dump({
                "instances": instances,
                "axis_align_matrix": axis_align_matrix.tolist()
            }, f)

    def process_frame(
            self,
            frame: str,
            lidar_objects: list,
            objects: dict,
            file: str,
            axis_align_matrix: np.ndarray,
    ) -> bool:
        """
        Process single frame, convert its objects to instances and store them.

        Args:
            frame (str):            current frame id
            lidar_object (list):    list of objects with lidar annotation
            objects (dict):         dict of all objects in frame
            file (str):             original file name
            axis_align_matrix:      4x4 axis align matrix

        Returns:
            bool True if success else False
        """
        if not lidar_objects:
            print_log(
                f"WARN: No lidar objects found in "
                f"{frame['frame_properties']['timestamp']} at "
                f"{osp.basename(file)}"
            )
            return False

        instances = self.convert_objects_in_frame(
            lidar_objects, frame, objects, file)

        if not self.dry_run:
            self.store_to_file(instances, frame, file, axis_align_matrix)
        return True

    def process_file(self, file: str) -> None:
        """
        Process single file and convert its frames parallel.

        Args:
            file (str):     path to file
        """
        data = self.load_label_file(file)

        objects = data["objects"]
        lidar_objects = self.extract_lidar_objects(objects)
        axis_align_matrix = self.compute_axis_aligned_matrix(
            data["coordinate_systems"]["lidar"])

        with futures.ThreadPoolExecutor(
                max_workers=self.max_workers
        ) as executor:
            threads = [
                executor.submit(
                    self.process_frame,
                    frame,
                    lidar_objects,
                    objects,
                    file,
                    axis_align_matrix,
                )
                for frame in data["frames"].values()
            ]
            for thread in futures.as_completed(threads):
                thread.result()


class OSDaR23PCDConverter:
    """
    Wrapper class to convert pointclouds from OSDaR23 into MMDetection3D
    binary format.

    Args:
        root_path (str):     path to raw data expecting */lidar/*.pcd
        out_dir (str):       root dir to store results (subdir points)
        max_workers (int):   max. workers for parallization (Default: 4)
        dry_run (bool):      convert without storing result
    """

    def __init__(
        self,
        root_path: str,
        out_dir: str,
        max_workers: int = 4,
        dry_run: bool = False,
    ):
        self.out_dir = out_dir
        self.files = glob.glob(f"{root_path}/*/lidar/*.pcd")
        self.max_workers = max_workers
        self.dry_run = dry_run

    def convert(self):
        """Wrapper for calling convertion for each measurement"""
        track_parallel_progress(
            self.process_file, self.files, nproc=self.max_workers)

    def process_file(self, file: str) -> None:
        """
        Wrapper for loading, converting and storing single file.

        Args:
            file (str):     path to file
        """
        pcd_data = _PCD.from_path(file)
        points = self.convert_pcd_data(pcd_data)
        self.store_data(points, file)

    def convert_pcd(self, pcd_data: dict) -> np.ndarray:
        """
        Converts point cloud into np.ndarray for binary storing.
        Uses dimension [x,y,z,intensity]

        Args:
            pcd_data (dict):        point cloud
            file (str):             original file name

        Returns:
            np.ndarray with four dimensions
        """
        n_pts = len(pcd_data.pc_data)
        points = np.zeros([n_pts, 4], dtype=np.float32)

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
    ) -> None:
        """
        Stors point cloud as binary file.

        Example of file naming:
            only first 3 numbers of frames
            orig filename: 22.1_xxx/81199.py
            filename: 22.1 | 0000 | 008119
                221 - Measurement
                0000 - Delimiter
                008119 - zero padded time in decaseconds

        Args:
            points (np.ndarray):    4d array with point cloud data
            file (str):             original file name
        """
        full_scene_number = osp.split(osp.dirname(osp.dirname(file)))[1]
        scene_number = full_scene_number.split('_')[-1].replace(".", "")
        frame_number = osp.basename(file)[:3]

        output_path = (
            f"{self.out_dir}/points/"
            f"{scene_number}0000{frame_number}.bin"
        )
        if not self.dry_run:
            with open(output_path, "wb") as f:
                f.write(points.tobytes())
                return output_path
        return None
