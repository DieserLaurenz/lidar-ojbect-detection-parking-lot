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
        "bbox": [-1.09, 12.58, 0.63, 4.7, 1.78, 1.64, 1.5707963267948966],
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
        "bbox": [15.94, 1.74, 0.73, 4.768, 1.74, 1.7, 1.5457963267948954],
        "num_pts": 327,
        "sensors": ["os1"],
    },
    {  # car lower right
        "class": "car",
        "bbox": [-1.9200000000000599, -5.619999999999993, 0.7249999999999999, 4.5, 1.9, 1.3499999999999999, 0.02079632679489812],
        "num_pts": 464,
        "sensors": ["os0", "os1"],
    },
]

CLASS_ALIASES = {
    "bicycle": "bike",
    "Cyclist": "bike",
    "Pedestrian": "person",
    "Car": "car",
}

CLASS_BOUNDS = {
    "car": {
        "xy_sorted": ([0.9, 2.2], [3.0, 6.5]),
        "z": [0.7, 2.5],
    },
    "bike": {
        "xy_sorted": ([0.25, 0.6], [0.8, 3.0]),
        "z": [0.5, 2.2],
    },
    "person": {
        "xy_sorted": ([0.15, 0.15], [1.2, 1.2]),
        "z": [0.7, 2.3],
    },
}


def normalize_class_name(class_name: str | None) -> str | None:
    if class_name is None:
        return None
    return CLASS_ALIASES.get(class_name, class_name)


def get_template_dim(class_name: str) -> list:
    class_name = normalize_class_name(class_name)
    for template in TEMPLATES:
        if template["name"] == class_name:
            return template["dim"]
    raise ValueError(f"Unknown class template: {class_name}")


def bbox_is_plausible_for_class(bbox: list, class_name: str) -> bool:
    class_name = normalize_class_name(class_name)
    bounds = CLASS_BOUNDS.get(class_name)
    if bounds is None:
        return True

    xy = sorted([bbox[3], bbox[4]])
    xy_min, xy_max = bounds["xy_sorted"]
    z_min, z_max = bounds["z"]
    return (
        xy_min[0] <= xy[0] <= xy_max[0] and
        xy_min[1] <= xy[1] <= xy_max[1] and
        z_min <= bbox[5] <= z_max
    )


def apply_template_dims(bbox: list, class_name: str) -> list:
    fixed_bbox = list(bbox)
    fixed_bbox[3:6] = get_template_dim(class_name)
    return fixed_bbox


def filter_points_by_roi(points: np.ndarray, roi: list | None) -> np.ndarray:
    if roi is None:
        return points
    x_min, x_max, y_min, y_max, z_min, z_max = roi
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    return points[mask]


def is_near_origin(bbox: list, radius: float) -> bool:
    if radius <= 0:
        return False
    return float(np.hypot(bbox[0], bbox[1])) < radius


def assign_class(bbox: list, filter: list = []) -> str:
    """
        Assign a bbox to a class by using globally defined TEMPLATES by the bbox dimension.

        Args:
            bbox: list   - bbox defined by [x, y, z, dx, dy, dz, yaw]
            filter: list - list of classes that should be ignored

        Returns:
            assigned_class: str - assigned class name
    """
    filter = [normalize_class_name(item) for item in filter]
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
    if not similarities:
        raise ValueError("No class left after applying filter_class")
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
        filter_class: list = [],
        include_static_labels: bool = True,
        expected_class: str | None = None,
        use_template_dims: bool = False,
        keep_largest: int = 0,
        bg_threshold: float = 0.05,
        max_foreground_ratio: float = 10.0,
        dbscan_eps: float = 0.99,
        min_cluster_points: int = 120,
        min_bbox_extent: float = 0.2,
        max_bbox_extent: float = 7.0,
        roi: list | None = None,
        exclude_origin_radius: float = 0.0,
) -> list:
    """
        Annotate pointcloud by background removal and static annotations.

        Args:
            background_path: str    - path to background pointcloud
            pcd_path: str           - path to current pointcloud
            vis: o3d.Visualization  - (Optional) o3d visualization where to add geometries
            filter_class: list      - list of class names to be ignored in assigning
            include_static_labels   - add hard-coded static boxes if True
            expected_class          - if set, only keep clusters for this class
            use_template_dims       - replace bbox dimensions by class template
            keep_largest            - keep only N largest dynamic clusters, 0 keeps all
            roi                     - optional [xmin, xmax, ymin, ymax, zmin, zmax]
            exclude_origin_radius   - discard clusters centered near the x/y origin

        Returns:
            bboxes: list            - list of tuple containing the annotations (bbox, class name, num_pts)
    """
    sensor_id = os.path.basename(os.path.dirname(pcd_path)).split("_")[0]

    background = o3d.io.read_point_cloud(background_path)
    pcd = o3d.io.read_point_cloud(pcd_path)

    voxel_size = 0.05
    pcd_bg_ds = background.voxel_down_sample(voxel_size)
    pcd_car_ds = pcd.voxel_down_sample(voxel_size)

    dists = np.asarray(pcd_car_ds.compute_point_cloud_distance(pcd_bg_ds))
    car_pts = np.asarray(pcd_car_ds.points)
    foreground_points = car_pts[dists > bg_threshold]
    foreground_points = filter_points_by_roi(foreground_points, roi).tolist()

    ratio = len(foreground_points) / len(pcd_car_ds.points) * 100
    print(f"Remaining {ratio:.0f}% of orig pcd")
    if ratio > max_foreground_ratio:
        print("Background Removal failed. Skipping...")
        foreground_points = []

    bboxes = []
    expected_class = normalize_class_name(expected_class)
    if include_static_labels:
        static_bboxes = add_static_bboxes(sensor_id, pcd)
        if expected_class is not None:
            static_bboxes = [
                item for item in static_bboxes
                if normalize_class_name(item[1]) == expected_class
            ]
        bboxes += static_bboxes

    if len(foreground_points):
        pcd_foreground = o3d.geometry.PointCloud()
        pcd_foreground.points = o3d.utility.Vector3dVector(
            np.array(foreground_points))

        labels = np.array(
            pcd_foreground.cluster_dbscan(
                eps=dbscan_eps,
                min_points=min_cluster_points,
            ))
        found_labels = labels[labels >= 0]
        if len(found_labels) == 0:
            print(f"No dynamic Objects found in {pcd_path}")
            return bboxes

        print(f"Found {len(np.bincount(found_labels))} cluster")

        cluster_ids = sorted(
            range(len(np.bincount(found_labels))),
            key=lambda cluster_id: int(np.sum(labels == cluster_id)),
            reverse=True,
        )
        accepted_dynamic = 0

        for idx in cluster_ids:
            points = np.asarray(
                pcd_foreground.points
            )[labels == idx]

            if len(points) < min_cluster_points:
                print("WARN: Cluster has to few points. Skipping")
                continue

            pcd_cluster_only = o3d.geometry.PointCloud()
            pcd_cluster_only.points = o3d.utility.Vector3dVector(points)

            bbox_obj, bbox_cdy = create_bbox_from_pcd(pcd_cluster_only)
            # discard unresonable extents
            if np.any(bbox_obj.extent[bbox_obj.extent > max_bbox_extent]):
                print("Discard bbox", bbox_cdy)
                continue
            if np.any(bbox_obj.extent[bbox_obj.extent < min_bbox_extent]):
                print("Discard bbox", bbox_cdy)
                continue

            bbox_class = expected_class or assign_class(bbox_cdy, filter=filter_class)
            if is_near_origin(bbox_cdy, exclude_origin_radius):
                print("Discard origin-near bbox", bbox_cdy)
                continue
            if expected_class is not None and not bbox_is_plausible_for_class(
                    bbox_cdy, expected_class):
                print("Discard bbox for expected class", expected_class, bbox_cdy)
                continue
            if use_template_dims:
                bbox_cdy = apply_template_dims(bbox_cdy, bbox_class)
                bbox_obj = create_bbox_from_label({
                    'label': bbox_class,
                    'bbox': bbox_cdy,
                })
            print(f"[DEBUG]: {bbox_class}, {bbox_cdy[3:6]}")

            bboxes.append(
                (bbox_cdy, bbox_class, len(pcd_cluster_only.points))
            )
            accepted_dynamic += 1

            if vis:
                vis.add_geometry(bbox_obj)
                vis.add_geometry(pcd_foreground)

            if keep_largest > 0 and accepted_dynamic >= keep_largest:
                break

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
    if isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox):
        bbox_center = bbox.get_center()
        bbox_extent = bbox.get_extent()
        yaw = 0.0
    else:
        bbox_center = bbox.center
        bbox_extent = bbox.extent
    bbox_label = bbox_center.tolist() + bbox_extent.tolist() + [yaw]

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
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    file, bg_frame, labels_path, dry_run, annotate_kwargs = args
    labels = annotate(bg_frame, file, **annotate_kwargs)
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
        include_static_labels: bool = True,
        expected_class: str | None = None,
        use_template_dims: bool = False,
        keep_largest: int = 0,
        bg_threshold: float = 0.05,
        max_foreground_ratio: float = 10.0,
        dbscan_eps: float = 0.99,
        min_cluster_points: int = 120,
        min_bbox_extent: float = 0.2,
        max_bbox_extent: float = 7.0,
        roi: list | None = None,
        exclude_origin_radius: float = 0.0,
) -> None:
    """Annotate given frames in parallel without visualization."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    files = load_pcd_names(pcd_dir, range)
    labels_path = os.path.join(os.path.dirname(pcd_dir), label_dir)
    os.makedirs(labels_path, exist_ok=True)

    annotate_kwargs = dict(
        filter_class=filter_class,
        include_static_labels=include_static_labels,
        expected_class=expected_class,
        use_template_dims=use_template_dims,
        keep_largest=keep_largest,
        bg_threshold=bg_threshold,
        max_foreground_ratio=max_foreground_ratio,
        dbscan_eps=dbscan_eps,
        min_cluster_points=min_cluster_points,
        min_bbox_extent=min_bbox_extent,
        max_bbox_extent=max_bbox_extent,
        roi=roi,
        exclude_origin_radius=exclude_origin_radius,
    )
    tasks = [(f, bg_frame, labels_path, dry_run, annotate_kwargs) for f in files]

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
        include_static_labels: bool = True,
        expected_class: str | None = None,
        use_template_dims: bool = False,
        keep_largest: int = 0,
        bg_threshold: float = 0.05,
        max_foreground_ratio: float = 10.0,
        dbscan_eps: float = 0.99,
        min_cluster_points: int = 120,
        min_bbox_extent: float = 0.2,
        max_bbox_extent: float = 7.0,
        roi: list | None = None,
        exclude_origin_radius: float = 0.0,
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
    global files, file_idx, running, is_replay, is_dry_run, labels_path, annotate_options
    files = load_pcd_names(pcd_dir, range)
    file_idx = 0
    running = True
    is_replay = replay
    is_dry_run = dry_run
    annotate_options = dict(
        filter_class=filter_class,
        include_static_labels=include_static_labels,
        expected_class=expected_class,
        use_template_dims=use_template_dims,
        keep_largest=keep_largest,
        bg_threshold=bg_threshold,
        max_foreground_ratio=max_foreground_ratio,
        dbscan_eps=dbscan_eps,
        min_cluster_points=min_cluster_points,
        min_bbox_extent=min_bbox_extent,
        max_bbox_extent=max_bbox_extent,
        roi=roi,
        exclude_origin_radius=exclude_origin_radius,
    )

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
        global running, files, file_idx, is_replay, is_dry_run, labels_path, annotate_options
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
                **annotate_options
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


def parse_roi(s):
    """
    Parse 'xmin,xmax,ymin,ymax,zmin,zmax' into a list of floats.
    """
    parts = s.split(",")
    if len(parts) != 6:
        raise ArgumentTypeError(
            "ROI must be 'xmin,xmax,ymin,ymax,zmin,zmax'")
    return [float(part) for part in parts]


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
        choices=["car", "bike", "bicycle", "person"],
        default=[],
        nargs="*",
    )
    parser.add_argument(
        "--expected-class",
        help="Only keep labels for the expected dynamic class.",
        choices=["car", "bike", "bicycle", "person"],
    )
    parser.add_argument(
        "--no-static-labels",
        help="Do not add hard-coded static parked-car labels.",
        action="store_true",
    )
    parser.add_argument(
        "--use-template-dims",
        help="Replace dynamic bbox dimensions by class template dimensions.",
        action="store_true",
    )
    parser.add_argument(
        "--keep-largest",
        type=int,
        default=0,
        help="Keep only the N largest dynamic clusters. 0 keeps all.",
    )
    parser.add_argument(
        "--bg-threshold",
        type=float,
        default=0.05,
        help="Distance threshold for background removal in meters.",
    )
    parser.add_argument(
        "--max-foreground-ratio",
        type=float,
        default=10.0,
        help="Skip dynamic labeling if foreground exceeds this percentage.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.99,
        help="DBSCAN epsilon for dynamic foreground clustering.",
    )
    parser.add_argument(
        "--min-cluster-points",
        type=int,
        default=120,
        help="Minimum points for DBSCAN clusters and retained clusters.",
    )
    parser.add_argument(
        "--min-bbox-extent",
        type=float,
        default=0.2,
        help="Discard boxes with any extent smaller than this.",
    )
    parser.add_argument(
        "--max-bbox-extent",
        type=float,
        default=7.0,
        help="Discard boxes with any extent larger than this.",
    )
    parser.add_argument(
        "--roi",
        type=parse_roi,
        help="Optional ROI as xmin,xmax,ymin,ymax,zmin,zmax.",
    )
    parser.add_argument(
        "--exclude-origin-radius",
        type=float,
        default=0.0,
        help="Discard dynamic clusters whose bbox center is this close to x/y origin.",
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
                include_static_labels=not args.no_static_labels,
                expected_class=args.expected_class,
                use_template_dims=args.use_template_dims,
                keep_largest=args.keep_largest,
                bg_threshold=args.bg_threshold,
                max_foreground_ratio=args.max_foreground_ratio,
                dbscan_eps=args.dbscan_eps,
                min_cluster_points=args.min_cluster_points,
                min_bbox_extent=args.min_bbox_extent,
                max_bbox_extent=args.max_bbox_extent,
                roi=args.roi,
                exclude_origin_radius=args.exclude_origin_radius,
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
                include_static_labels=not args.no_static_labels,
                expected_class=args.expected_class,
                use_template_dims=args.use_template_dims,
                keep_largest=args.keep_largest,
                bg_threshold=args.bg_threshold,
                max_foreground_ratio=args.max_foreground_ratio,
                dbscan_eps=args.dbscan_eps,
                min_cluster_points=args.min_cluster_points,
                min_bbox_extent=args.min_bbox_extent,
                max_bbox_extent=args.max_bbox_extent,
                roi=args.roi,
                exclude_origin_radius=args.exclude_origin_radius,
            )
