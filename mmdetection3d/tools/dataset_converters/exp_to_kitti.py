import numpy as np
import os
import glob
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Transformations
# KITTI looks in X direction, Y to the left, Z is up
THETA = np.deg2rad(90)
c, s = np.cos(THETA), np.sin(THETA)
R = np.array([[c, -s, 0],
              [s,  c, 0],
              [0,  0, 1]], dtype=np.float32)

T = np.array([30, 0.0, -3.3], dtype=np.float32)

CLASS_MAP_LUMPI_TO_KITTI = {
    0: 0,  # person - Pedestrian
    1: 2,  # car - Car
    2: 1,  # bicycle - Cyclist
}


def _process_file(
        bin_path: str,
        output_dir: str,
        remove_ceiling: bool = False,
        intensity_norm_max: float = 1.0
) -> None:
    """
    For single file apply Transformation from Experiment World into
    KITTI world on pointcloud data.

    Args:
        input_dir: str              - path to dir of original pointclouds
        output_dir: str             - path to dir where to store
                                      transformed pointclouds
        remove_ceiling: bool        - whether to remove the ceiling from
                                      the pointclouds
        intensity_norm_max: float   - normalize intensities values
                                      between 0 and ...
    """
    fname = os.path.basename(bin_path)

    out_path = os.path.join(output_dir, fname)
    if os.path.exists(out_path):
        return

    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    # Remove ceiling before transform
    if remove_ceiling:
        no_ceiling_mask = pts[:, 2] < 1.9
        pts = pts[no_ceiling_mask]

    pts[:, :3] = pts[:, :3] @ R.T
    pts[:, :3] += T

    intensities = pts[:, 3]
    intensities[~np.isfinite(intensities)] = 1.0
    pts[:, 3] = (
        (intensities + np.min(intensities)) * intensity_norm_max /
        np.max(intensities)
    )

    out_path = os.path.join(output_dir, fname)
    pts.astype(np.float32).tofile(out_path)


def convert_pointclouds(
        input_dir: str,
        output_dir: str,
        remove_ceiling: bool = False,
        intensity_norm_max: float = 1.0,
        max_workers: int = 8,
) -> None:
    """
    Apply Transformation from Experiment World into KITTI world on
    pointcloud data.

    Args:
        input_dir: str              - path to dir of original pointclouds
        output_dir: str             - path to dir where to store transformed
                                      pointclouds
        remove_ceiling: bool        - whether to remove the ceiling from the
                                      pointclouds
        intensity_norm_max: float   - normalize intensities values
                                      between 0 and ...
        max_workers: int            - parallize with Processes wit a maximum of
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*.bin")))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_file,
                file,
                output_dir,
                remove_ceiling=remove_ceiling,
                intensity_norm_max=intensity_norm_max,
            )
            for file in files
        ]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass


def convert_labels(input_dir: str, output_dir: str) -> None:
    """
        Apply Transformation from Experiment World into KITTI world on labels.

        Args:
            input_dir: str  - path to dir of original label
            output_dir: str - path to dir where to store transformed labels
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    for label_path in tqdm(files):
        fname = os.path.basename(label_path)
        out_path = os.path.join(output_dir, fname)
        if os.path.exists(out_path):
            continue

        with open(label_path, "r") as fp:
            label = json.load(fp)
            assert (
                isinstance(label, dict) and
                "instances" in label and
                isinstance(label['instances'], list)
            ), f"{fname} does not contain 'instances' list"

            for instance in label["instances"]:
                bbox = np.asarray(instance["bbox_3d"], dtype=np.float32)
                bbox[:3] = R @ bbox[:3] + T
                bbox[6] += THETA
                # Normalize heading to [-pi, pi]
                bbox[6] = (bbox[6] + np.pi) % (2*np.pi) - np.pi
                instance["bbox_3d"] = bbox.tolist()

                new_label = CLASS_MAP_LUMPI_TO_KITTI[instance['bbox_label_3d']]
                instance['bbox_label_3d'] = new_label

            with open(out_path, "w") as wp:
                json.dump(label, wp)


if __name__ == "__main__":
    input_dir = "data/exp/points"
    output_dir = "data/exp/points_kitti"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Convert Pointclouds from {input_dir} to {output_dir}")
    convert_pointclouds(input_dir, output_dir)

    input_dir = "data/exp/labels"
    output_dir = "data/exp/labels_kitti"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Convert Labels from {input_dir} to {output_dir}")
    convert_labels(input_dir, output_dir)
