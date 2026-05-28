import glob
import numpy as np
import open3d as o3d
import time
import csv
import os

from pypcd4 import PointCloud
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_intensities(path: str) -> list:
    """
        Loads intensity from pointcloud.
        Workaround because open3d does not load intensities.

        Args:
            path: str - path to pointcloud

        Returns:
            intensities: list - list of intensities to every point in pointclouds
    """
    pc = PointCloud.from_path(path)
    intensities = pc.pc_data['intensity']

    return intensities


def pack_colors(pcd_colors: list) -> np.ndarray:
    """
        Convert point cloud color for storing.

        Args:
            pcd_colors: list    - list of current color of every point in pointcloud

        Returns:
            rgb_float: list     - list of packed color of every point in pointcoud
    """
    colors = (np.asarray(pcd_colors) * 255).astype(np.uint8)
    r = colors[:, 0].astype(np.uint32)
    g = colors[:, 1].astype(np.uint32)
    b = colors[:, 2].astype(np.uint32)

    # Pack RGB into a single 32-bit integer
    rgb_int = (r << 16) | (g << 8) | b
    # reinterpret as float32 for PCD
    rgb_float = rgb_int.view(np.float32)

    return rgb_float


def store_merge_time(start_time: float, end_time: float, experiment_id: str, csv_path: str) -> None:
    """
        Add time metric for current merge to csv file as metric.
        Create file and add header if it does not exists.

        Args:
            start_time: float   - timestamp merge started
            end_time: floa      - timestampe merge ended
            experiment_id: str  - experiment identifier
            csv_path: str       - path to store metric
    """
    duration = end_time - start_time
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "experiment", "duration_seconds"])
        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                experiment_id,
                f"{duration:.6f}"
            ])


def store_pcds(
        pcd1: o3d.geometry.PointCloud,
        pcd2: o3d.geometry.PointCloud,
        merged: o3d.geometry.PointCloud,
        pcd1_intensities: np.ndarray,
        pcd2_intensities: np.ndarray,
        pcd1_path: str,
        pcd2_path: str,
        base_dir1: str,
        merged_dir: str,
        transform_dir1: str,
        transform_dir2: str,
) -> None:
    """
        Pack and Store merged and transformed source pointclouds.

        Args:
            pcd1: o3d.geometry.PointCloud   - transformed pointcloud of first sensor
            pcd2: o3d.geometry.PointCloud   - transformed pointcloud of second sensor
            merged: o3d.geometry.PointCloud - transformed and merged pointcloud of both sensors
            pcd1_intensities: np.ndarray    - intensities for first pointcloud
            pcd2_intensities: np.ndarray    - intensities for second pointcloud
            pcd1_path: str                  - path to first pointcloud
            pcd2_path: str                  - path to second pointcloud
            merged_dir: str                 - path of dir to store the merged pointclouds
            transform_dir1: str             - path of dir to store the transformed first pointcloud
            transform_dir2: str             - path of dir to store the transformed second pointcloud
    """
    merged_intensities = np.concatenate(
        [pcd1_intensities, pcd2_intensities],
        axis=0,
    )
    merged_points = np.asarray(merged.points).reshape(-1, 3)

    points_xyzirgb = np.zeros((merged_points.shape[0], 5), dtype=np.float32)
    points_xyzirgb[:, 0:3] = merged_points
    points_xyzirgb[:, 3] = merged_intensities
    points_xyzirgb[:, 4] = pack_colors(merged.colors)

    # Store merged pcd
    filename = f"{os.path.basename(pcd1_path).split('.')[0]}.pcd"
    path = os.path.join(base_dir1, merged_dir, filename)
    merged_pcd = PointCloud.from_xyzirgb_points(points_xyzirgb)
    merged_pcd.save(path)

    filename1 = os.path.basename(pcd1_path)
    filename2 = os.path.basename(pcd2_path)
    pcd1_points = np.asarray(pcd1.points)
    pcd2_points = np.asarray(pcd2.points)

    # Store transform pcd for sensor os0
    points1_xyzirgb = np.zeros((pcd1_points.shape[0], 5), dtype=np.float32)
    points1_xyzirgb[:, 0:3] = pcd1_points
    points1_xyzirgb[:, 3] = pcd1_intensities
    points1_xyzirgb[:, 4] = pack_colors(pcd1.colors)
    pcd1_transf = PointCloud.from_xyzirgb_points(points1_xyzirgb)
    pcd1_transf.save(os.path.join(transform_dir1, filename1))

    # Store transform pcd for sensor os1
    points2_xyzirgb = np.zeros((pcd2_points.shape[0], 5), dtype=np.float32)
    points2_xyzirgb[:, 0:3] = pcd2_points
    points2_xyzirgb[:, 3] = pcd2_intensities
    points2_xyzirgb[:, 4] = pack_colors(pcd2.colors)
    pcd2_transf = PointCloud.from_xyzirgb_points(points2_xyzirgb)
    pcd2_transf.save(os.path.join(transform_dir2, filename2))


def merge_pcds(
        pcd1_path: str,
        pcd2_path: str,
        show: bool = False,
        merged_dir: str = "merged_pcd",
        transform_dir: str = "transform",
        csv_path: str = "merge_times.csv",
        dry_run: bool = False,
) -> None:
    """
        Merge two pointclouds by applying manual transformation first and use ICP.
        The transformated source point clouds are also stroed.

        Args:
            pcd1_path: str  - path to first pointcloud
            pcd1_path: str  - path to second pointcloud
            show: bool      - if True open3d visual is shown
            merged_dir: str - in which dir the merged pointcloud should be stored
            transform_dir   - in which dir the transformed source pointclouds should be stored.
            csv_path: str   - path for csv file to store merge time metric
            dry_run: str    - if True now file process does not create any file
    """
    base_dir1 = os.path.dirname(os.path.dirname(pcd1_path))
    base_dir2 = os.path.dirname(os.path.dirname(pcd1_path))
    assert base_dir1 == base_dir2, (
        f"Failed to extract base dir from {pcd1_path} and {pcd2_path}")

    transform_dir1 = os.path.join(base_dir1, f"os0_pcd_{transform_dir}")
    transform_dir2 = os.path.join(base_dir2, f"os1_pcd_{transform_dir}")
    os.makedirs(transform_dir1, exist_ok=True)
    os.makedirs(transform_dir2, exist_ok=True)
    os.makedirs(os.path.join(base_dir1, merged_dir), exist_ok=True)

    start_time = time.perf_counter()
    pcd1 = o3d.io.read_point_cloud(pcd1_path)
    pcd2 = o3d.io.read_point_cloud(pcd2_path)
    pcd1_intensities = get_intensities(pcd1_path)
    pcd2_intensities = get_intensities(pcd2_path)

    pcd1.remove_non_finite_points()
    pcd2.remove_non_finite_points()
    pcd1, ind1 = pcd1.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
    pcd2, ind2 = pcd2.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)

    pcd1_intensities = pcd1_intensities[ind1]
    pcd2_intensities = pcd2_intensities[ind2]

    # gathered by manual transforming in blender
    T_pcd1 = np.array((
        (0.8853409886360168, -0.46335849165916443, 0.03834317624568939, 0.0),
        (0.4648183286190033, 0.8801859617233276, -0.09600342810153961, 0.0),
        (0.010734880343079567, 0.10281837731599808,
         0.9946421980857849, 1.9199999570846558),
        (0.0, 0.0, 0.0, 1.0)
    ))
    T_pcd2 = np.array((
        (-0.017428433522582054, 0.9998477101325989, -
         0.0009133866406045854, 3.0199997425079346),
        (-0.9984774589538574, -0.01745235174894333, -
         0.05232798680663109, 11.139999389648438),
        (-0.0523359589278698, 1.1032486035844613e-09,
         0.9986295104026794, 1.0600000143051147),
        (0.0, 0.0, 0.0, 1.0)
    ))
    pcd1.transform(T_pcd1)
    pcd2.transform(T_pcd2)

    voxel_size = 0.05
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    pcd2_down = pcd2.voxel_down_sample(voxel_size)

    pcd1_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30)
    )
    pcd2_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30)
    )

    result_icp = o3d.pipelines.registration.registration_icp(
        pcd1_down, pcd2_down, voxel_size * 1.5, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    pcd1.transform(result_icp.transformation)

    pcd1.paint_uniform_color([1, 0, 0])  # Red
    pcd2.paint_uniform_color([0, 1, 0])  # Green

    merged = pcd1 + pcd2

    end_time = time.perf_counter()
    experiment = os.path.dirname(os.path.dirname(pcd1_path))
    store_merge_time(start_time, end_time, experiment, csv_path)

    if not dry_run:
        store_pcds(
            pcd1,
            pcd2,
            merged,
            pcd1_intensities,
            pcd2_intensities,
            pcd1_path,
            pcd2_path,
            base_dir1,
            merged_dir,
            transform_dir1,
            transform_dir2,
        )

    if show:
        lookat = [
            3.3221172409870863,
            0.87555069656852125,
            -1.0175931695237883
        ]
        front = [
            0.045677265956678564,
            -0.97562757025918767,
            0.21462626010084762
        ]
        up = [
            -0.025067432678240169,
            0.21366341459708788,
            0.97658566909495192
        ]
        zoom = 0.3
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        o3d.visualization.draw_geometries([merged, frame],
                                          zoom=zoom,
                                          front=front,
                                          lookat=lookat,
                                          up=up,
                                          )


def worker(pair: tuple, merged_dir: str, csv_path: str, dry_run: bool) -> None:
    """
        Wrapper for multi-threaded run call

        Args:
            pair: tuple     - contains path to both pcd file for merging
            csv_path: str   - path for csv file to store merge time metric
            dry_run: str    - if True now file process does not create any file
    """
    pcd1, pcd2 = pair
    merge_pcds(
        pcd1,
        pcd2,
        show=False,
        merged_dir=merged_dir,
        csv_path=csv_path,
        dry_run=dry_run
    )


def run(
        src_dir1: str,
        src_dir2: str,
        show: bool,
        merged_dir: str,
        csv_path: str,
        dry_run: bool,
        max_worker: int
) -> None:
    """
        Wrapper for starting from the main function

        Args:
            src_dir1: str   - source dir of sensor os0
            src_dir2: str   - source dir of sensor os1
            show: bool      - if open3d should visualize results
            merged_dir: str - path to dir to store merged point clouds
            csv_path: str   - path for csv file to store merge time metric
            dry_run: str    - if True now file process does not create any file
            max_worker: int - how many threads are created. Does not apply if show is True
    """
    pcd1_files = glob.glob(os.path.join(src_dir1, "*.pcd"))
    pcd2_files = glob.glob(os.path.join(src_dir2, "*.pcd"))

    pairs = zip(pcd1_files, pcd2_files)
    if show:
        for pcd1, pcd2 in tqdm(pairs, total=len(pcd1_files), unit="pcd"):
            merge_pcds(
                pcd1,
                pcd2,
                show=show,
                merged_dir=merged_dir,
                csv_path=csv_path,
                dry_run=dry_run
            )
    else:
        with ThreadPoolExecutor(max_workers=max_worker) as executor:
            futures = [
                executor.submit(worker,
                                pair,
                                merged_dir,
                                csv_path,
                                dry_run)
                for pair in pairs
            ]

            for _ in tqdm(as_completed(futures), total=len(futures), unit="pcd"):
                pass


if __name__ == "__main__":
    if os.name == "posix":
        # force to use x11 on wayland
        os.environ["XDG_SESSION_TYPE"] = "x11"

    parser = ArgumentParser(
        description="Merge all point clouds from two sensors.")
    parser.add_argument("src_dir1", help="Path to dir of pcds 1")
    parser.add_argument("src_dir2", help="Path to dir of pcds 2")
    parser.add_argument(
        "--show", help="Show each merged pcd", action="store_true")
    parser.add_argument(
        "--merged-dir",
        help="Name of merged_pcd in base dir",
        default="merged_pcd",
    )
    parser.add_argument(
        "--dry-run",
        help="Dry run do not save pcd files",
        action="store_true",
    )
    parser.add_argument(
        "--csv-path",
        help="Path for csv for storing merge times",
        default="merge_times.csv",
    )
    parser.add_argument(
        "--max-worker",
        type=int,
        help="Number of max. worker for parallel processing. If show is True only single threaded!",
        default="8",
    )
    args = parser.parse_args()

    run(
        args.src_dir1,
        args.src_dir2,
        show=args.show,
        merged_dir=args.merged_dir,
        csv_path=args.csv_path,
        dry_run=args.dry_run,
        max_worker=args.max_worker,
    )
