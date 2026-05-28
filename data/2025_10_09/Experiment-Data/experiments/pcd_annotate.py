import os
import json

from argparse import ArgumentParser, ArgumentTypeError
from glob import glob
import open3d as o3d
import numpy as np

TEMPLATES = [
    {
        "name": "car",
        "dim": [4.5, 1.8, 1.5],
    },
    {
        "name": "person",
        "dim": [0.5, 0.5, 1.5],
    },
    {
        "name": "bike",
        "dim": [2.0, 0.6, 1.1],
    },
]

STATIC_LABELS = [
    {  # car back
        "class": "car",
        "bbox": [-1.09, 12.58, 0.63, 1.78, 4.7, 1.64, 0],
        "num_pts": 8328,
        "sensors": ["os0", "os1"],
    },
    {  # Van
        "class": "car",
        "bbox": [-1.15, 1.106, 0.91, 5.32, 1.84, 1.84, 1.972222054],
        "num_pts": 22813,
        "sensors": ["os1"],
    },
    {  # car back 2
        "class": "car",
        "bbox": [-2.27, 19.69, 0.56, 3.96, 1.52, 1.42, 0],
        "num_pts": 575,
        "sensors": ["os1"],
    },
    {  # car far right
        "class": "car",
        "bbox": [15.84, 1.74, 0.73, 1.74, 4.468, 1.7, 0],
        "num_pts": 425,
        "sensors": ["os1"],
    },
]


def assign_class(bbox: list, filter: list = []) -> str:
    """
        Assign a bbox to a class by using globally defined TEMPLATES by the bbox dimension.

        Args:
            bbox: list   - bbox defined by [x, y, z, dx, dy, dz, yaw]
            filter: list - list of classes that should be ignored

        Returns:
            assigned_class: str - assigned class name
    """
    similarities = {}
    for temp in TEMPLATES:
        name, dim = temp.values()
        if name in filter:
            continue
        if len(dim) != 3:
            print("Error: Invalid TEMPLATE")
            print(temp)
            continue
        dl = abs(bbox[3] - dim[0])
        dw = abs(bbox[4] - dim[1])
        dh = abs(bbox[5] - dim[2])
        similarity = dl + dw + dh
        similarities[name] = similarity
    assigned_class = min(similarities, key=lambda k: similarities[k])
    return assigned_class


def is_bbox_in_pcd(bbox: list, pcd: o3d.geometry.PointCloud, thr: int = 120) -> bool:
    """
        Tests if enough points of the pointcloud are inside the bbox.

        Args:
            bbox: list                   - bbox defined by [x, y, z, dx, dy, dz, yaw]
            pcd: o3d.geometry.PointCloud - pointcloud to test with
            thr: int                     - threshold at which number of points the result is True

        Returns:
            len(mask) > thre             - Boolean if number of points is greater than threshold
    """
    cx, cy, cz, dx, dy, dz, yaw = bbox
    center = np.array([cx, cy, cz])
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    obb = o3d.geometry.OrientedBoundingBox(center, R, [dx, dy, dz])
    mask = obb.get_point_indices_within_bounding_box(pcd.points)

    return len(mask) > thr


def add_static_bboxes(sensor_id: str, pcd: o3d.geometry.PointCloud, thr: float = 0.5) -> list:
    """
        Create a list of annotations if more than 50% of total points are inside the bbox.

        Args:
            sensor_id: str               - sensor_id to filter static labels
            pcd: o3d.geometry.PointCloud - current pointcloud to test with
            thr: float                   - percentage of the original num_pts needs to be inside

        Returns:
            bboxes: list                 - list of tuple annoations containing (bbox, class name, num_pts)
    """
    bboxes = []
    for bbox in STATIC_LABELS:
        if not is_bbox_in_pcd(bbox['bbox'], pcd, thr=bbox['num_pts'] * thr):
            continue
        if sensor_id == "merged":
            bboxes.append((bbox['bbox'], bbox['class'], bbox['num_pts']))
        else:
            if sensor_id in bbox['sensors']:
                bboxes.append((bbox['bbox'], bbox['class'], bbox['num_pts']))

    print(f"Add {len(bboxes)} Static Labels")
    return bboxes


def annotate(
        background_path: str,
        pcd_path: str,
        vis=None,
        filter_class: list = []
) -> list:
    """
        Annotate pointcloud by background removal and static annotations.

        Args:
            background_path: str    - path to background pointcloud
            pcd_path: str           - path to current pointcloud
            vis: o3d.Visualization  - (Optional) o3d visualization where to add geometries
            filter_class: list      - list of class names to be ignored in assigning

        Returns:
            bboxes: list            - list of tuple containing the annotations (bbox, class name, num_pts)
    """
    sensor_id = os.path.basename(os.path.dirname(pcd_path)).split("_")[0]

    background = o3d.io.read_point_cloud(background_path)
    pcd = o3d.io.read_point_cloud(pcd_path)

    voxel_size = 0.05
    pcd_bg_ds = background.voxel_down_sample(voxel_size)
    pcd_car_ds = pcd.voxel_down_sample(voxel_size)

    threshold = 0.05
    dists = np.asarray(pcd_car_ds.compute_point_cloud_distance(pcd_bg_ds))
    car_pts = np.asarray(pcd_car_ds.points)
    foreground_points = car_pts[dists > threshold].tolist()

    ratio = len(foreground_points) / len(pcd_car_ds.points) * 100
    print(f"Remaining {ratio:.0f}% of orig pcd")
    if ratio > 10:
        print("Background Removal failed. Skipping...")
        foreground_points = []

    bboxes = []
    bboxes += add_static_bboxes(sensor_id, pcd)

    if len(foreground_points):
        pcd_foreground = o3d.geometry.PointCloud()
        pcd_foreground.points = o3d.utility.Vector3dVector(
            np.array(foreground_points))

        labels = np.array(
            pcd_foreground.cluster_dbscan(eps=0.99, min_points=120))
        found_labels = labels[labels >= 0]
        if len(found_labels) == 0:
            print(f"No dynamic Objects found in {pcd_path}")
            return bboxes

        print(f"Found {len(np.bincount(found_labels))} cluster")

        for idx in range(len(np.bincount(found_labels))):
            points = np.asarray(
                pcd_foreground.points
            )[labels == idx]

            if len(points) < 4:
                print("WARN: Cluster has to few points. Skipping")
                continue

            pcd_cluster_only = o3d.geometry.PointCloud()
            pcd_cluster_only.points = o3d.utility.Vector3dVector(points)

            bbox_obj, bbox_cdy = create_bbox_from_pcd(pcd_cluster_only)
            # discard unresonable extents
            if np.any(bbox_obj.extent[bbox_obj.extent > 7.0]):
                print("Discard bbox", bbox_cdy)
                continue
            if np.any(bbox_obj.extent[bbox_obj.extent < 0.2]):
                print("Discard bbox", bbox_cdy)
                continue

            bbox_class = assign_class(bbox_cdy, filter=filter_class)
            print(f"[DEBUG]: {bbox_class}, {bbox_cdy[3:6]}")

            bboxes.append(
                (bbox_cdy, bbox_class, len(pcd_cluster_only.points))
            )

            if vis:
                vis.add_geometry(bbox_obj)
                vis.add_geometry(pcd_foreground)

    return bboxes


def create_bbox_from_label(label: dict) -> o3d.geometry.OrientedBoundingBox:
    """
        Create a o3d bbox object by label dict.

        Args:
            label: dict     - dict with key bbox requried (list of [x,y,z,dx,dy,dz,yaw])

        Returns:
            bbox_obj: o3d.geometry.OrientedBoundingBox
    """
    center = label['bbox'][0:3]
    extent = label['bbox'][3:6]
    yaw = label['bbox'][6]

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R_yaw = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])

    bbox_obj = o3d.geometry.OrientedBoundingBox(center, R_yaw, extent)
    bbox_obj.color = (0, 0, 1)

    return bbox_obj


def create_bbox_from_pcd(pcd: o3d.geometry.PointCloud) -> tuple:
    """
        Create tuple of bbox (7 elements) and o3d bbox object from Pointcloud.

        Args:
            pcd: o3d.geometry.PointCloud                        - source pointcloud

        Returns:
            bbox_obj: o3d.geometry.PointCloud, bbox_label: list - tuple of o3d bbox object and 7 elements bbox label
    """
    obb = pcd.get_oriented_bounding_box()
    R = obb.R
    yaw = np.arctan2(R[1, 0], R[0, 0])
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R_yaw = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])
    bbox = o3d.geometry.OrientedBoundingBox(obb.center, R_yaw, obb.extent)
    num_pts_bbox = len(bbox.get_point_indices_within_bounding_box(pcd.points))

    # Dirty fallback fix for e.g. Persons
    if num_pts_bbox < len(pcd.points) * 0.8:
        bbox = pcd.get_axis_aligned_bounding_box()

    bbox_obj = bbox.get_minimal_oriented_bounding_box()
    bbox_obj.color = (0, 0, 1)
    bbox_label = bbox.center.tolist() + bbox.extent.tolist() + [yaw]

    return bbox_obj, bbox_label


def store_label(annotation: list, file: str, labels_path: str) -> None:
    """
        Store annotation to json file.

        Args:
            annotation: list    - list of dicts containing all annotation for this frame
            file: str           - file path of the current pointcloud
            labels_path: str    - path to dir where to store the annotations

    """
    if not annotation:
        print("WARN: Annotation is empty. Skip storing")
        return

    file_name = os.path.basename(file).split('.')[0]
    label_name = f"{file_name}.json"
    path = os.path.join(labels_path, label_name)
    with open(path, "w") as fp:
        print(f"Stored label to {path}")
        json.dump(annotation, fp)


def load_label(file: str, labels_path: str) -> list:
    """
        Load annotation list from file. If file does not exit return empty list.

        Args:
            file:       - file path of the current pointcloud
            labels_path - path where labels are stored

        Returns:
            labels: list - list of dicts containing all annotation for this frame
    """
    label_name = os.path.basename(file).split('.')[0]
    label_file = os.path.join(labels_path, f"{label_name}.json")
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            labels = json.load(f)
            return labels

    return []


def load_pcd_names(pcd_dir: str, range: tuple) -> list:
    """
        Load all pcd file in given dir and filter by range.

        Args:
            pcd_dir: str    - path to pcd dir
            range: tuple    - tuple of list index a[range[0]:range[1]]

        Returns:
            files           - list of filtered file names in dir
    """
    files = sorted(glob(os.path.join(pcd_dir, "*.pcd")))[range[0]:range[1]]

    assert files, f"Error: Could not find any .pcd file at {pcd_dir}"
    return files


def _annotate_one(args):
    # Limit internal thread pools per worker to avoid oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    file, bg_frame, labels_path, dry_run, filter_class = args
    labels = annotate(bg_frame, file, filter_class=filter_class)
    annotations = []
    for (bbox, label, num_pts) in labels:
        if label is not None:
            annotations.append({
                'label': label,
                'bbox': bbox,
                'num_lidar_pts': num_pts,
            })
    if not dry_run:
        store_label(annotations, file, labels_path)
    return file


def auto_annotate_without_vis(
        pcd_dir: str,
        bg_frame: str,
        label_dir: str = "labels",
        range: tuple = (None, None),
        dry_run: bool = False,
        filter_class: list = [],
        workers: int = 1,
) -> None:
    """Annotate given frames in parallel without visualization."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    files = load_pcd_names(pcd_dir, range)
    labels_path = os.path.join(os.path.dirname(pcd_dir), label_dir)
    os.makedirs(labels_path, exist_ok=True)

    tasks = [(f, bg_frame, labels_path, dry_run, filter_class) for f in files]

    if workers <= 1:
        for t in tqdm(tasks, unit="frame"):
            _annotate_one(t)
        return

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = [ex.submit(_annotate_one, t) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), unit="frame"):
            pass


def auto_annotate_with_vis(
        pcd_dir: str,
        bg_frame: str,
        label_dir: str = "labels",
        replay: bool = False,
        range: tuple = (None, None),
        dry_run: bool = False,
        filter_class: list = [],
) -> None:
    """
        Annotate given frame with o3d visualization. Use Space to toggle autoplay to next frame!

        Args:
            pcd_dir: str        - path to pcd dir
            bg_frame: str       - path to background pcd to be used for background removal
            label_dir: str      - path to dir where annotations should be stored
            range: tuple        - tuple of list index a[range[0]:range[1]]
            dry_run: bool       - if True no file is written
            filter_class: list  - list of class names to ignore in this frame
    """
    global files, file_idx, running, is_replay, is_dry_run, labels_path, class_filtered
    files = load_pcd_names(pcd_dir, range)
    file_idx = 0
    running = True
    is_replay = replay
    is_dry_run = dry_run
    class_filtered = filter_class

    labels_path = os.path.join(
        os.path.dirname(pcd_dir),
        label_dir
    )
    os.makedirs(labels_path, exist_ok=True)

    def key_action_callback(vis, action, mods):
        global running

        # 0 down, 1 up, 2 repeat
        if action != 1:
            return True

        running = not running
        print(f"Playback {'started' if running else 'stopped'} ")

        return True

    def animation_callback(vis):
        global running, files, file_idx, is_replay, is_dry_run, labels_path, class_filtered
        if not running or file_idx > len(files) - 1:
            return

        params = vis.get_view_control().convert_to_pinhole_camera_parameters()

        print(f"{'-' * 5} [{file_idx}/{len(files) - 1}] {'-' * 5}")

        vis.clear_geometries()
        vis.add_geometry(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
        )

        if is_replay:
            labels = load_label(files[file_idx], labels_path)
            for label in labels:
                vis.add_geometry(create_bbox_from_label(label))
            print(f"Loaded {len(labels)} Labels")
            vis.add_geometry(o3d.io.read_point_cloud(files[file_idx]))
        else:
            labels = annotate(
                bg_frame,
                files[file_idx],
                vis=vis,
                filter_class=class_filtered
            )
            annotations = []
            for (bbox, label, num_pts) in labels:
                annotations.append({
                    'label': label,
                    'bbox': bbox,
                    'num_lidar_pts': num_pts,
                })
            if not is_dry_run:
                store_label(annotations, files[file_idx], labels_path)

        vis.get_view_control().convert_from_pinhole_camera_parameters(params)

        if file_idx == 0:
            ctr = vis.get_view_control()
            ctr.set_front([0.1, -0.8, 0.6])
            ctr.set_lookat([3.2, 2.8, -2.5])
            ctr.set_up([0.0, 0.6, 0.8])
            ctr.set_zoom(0.3)

        file_idx += 1

        return False

    SPACE = 32
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_action_callback(SPACE, key_action_callback)
    vis.register_animation_callback(animation_callback)

    vis.create_window()
    vis.add_geometry(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    )
    vis.run()


def parse_range(s):
    """
    Parse a string like 'start:end' or 'start:' or ':end' into a tuple (start, end).
    """
    parts = s.split(":")
    if len(parts) != 2:
        raise ArgumentTypeError(f"Invalid range format: '{s}'")

    start, end = parts
    start_val = int(start) if start else None
    end_val = int(end) if end else None
    return (start_val, end_val)


if __name__ == "__main__":
    if os.name == "posix":
        # force to use x11 on wayland
        os.environ["XDG_SESSION_TYPE"] = "x11"

    parser = ArgumentParser(
        description="Try to automatically annotate merged PCD using background removal"
    )
    parser.add_argument("pcd_dir", help="Path to dir of pcds")
    parser.add_argument(
        "--label-dir",
        help="Name of directory to store labels (inside pcd_dir)",
        required=True,
    )
    parser.add_argument(
        "--bg-frame",
        help="Path to pcd background frame that should be used for filtering",
    )
    parser.add_argument(
        "--result-only",
        help="just replay pcd with labels instead of creating",
        action="store_true"
    )
    parser.add_argument(
        "--show",
        help="Display every frame when annotation",
        action="store_true"
    )
    parser.add_argument(
        "--dry-run",
        help="Do not store labels. Useful for debugging with --show",
        action="store_true",
    )
    parser.add_argument(
        "--range",
        help=(
            "Select used frames, e.g. `100:`. Useful for debbuging "
            "with --show and --dry-run if the first frames are empty"
        ),
        type=parse_range,
        default=":",
    )
    parser.add_argument(
        "--filter-class",
        help="Class to ignore when assigning.",
        choices=["car", "bicycle", "person"],
        default=[],
        nargs="*",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Number of parallel worker processes (only used without --show).",
    )

    args = parser.parse_args()

    if args.result_only:
        auto_annotate_with_vis(
            args.pcd_dir,
            args.bg_frame,
            label_dir=args.label_dir,
            replay=True,
            range=args.range,
            dry_run=args.dry_run,
        )
    else:
        if args.show:
            auto_annotate_with_vis(
                args.pcd_dir,
                args.bg_frame,
                label_dir=args.label_dir,
                range=args.range,
                dry_run=args.dry_run,
                filter_class=args.filter_class,
            )
        else:
            auto_annotate_without_vis(
                args.pcd_dir,
                args.bg_frame,
                label_dir=args.label_dir,
                range=args.range,
                dry_run=args.dry_run,
                filter_class=args.filter_class,
                workers=args.workers,
            )
