import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "2025_10_09"
    / "Experiment-Data"
    / "experiments"
)
IGNORE_FRAME_KEY = "ignore_frame"


def is_ignore_frame_annotation(data: object) -> bool:
    if isinstance(data, dict):
        return bool(data.get(IGNORE_FRAME_KEY, False))
    if isinstance(data, list):
        return any(
            bool(item.get(IGNORE_FRAME_KEY, False))
            for item in data
            if isinstance(item, dict)
        )
    return False


def read_pcd_xyz(path: Path) -> np.ndarray:
    fields = None
    sizes = None
    types = None
    counts = None
    points = None

    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Missing DATA line in {path}")
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            if key == "FIELDS":
                fields = values
            elif key == "SIZE":
                sizes = [int(v) for v in values]
            elif key == "TYPE":
                types = values
            elif key == "COUNT":
                counts = [int(v) for v in values]
            elif key == "POINTS":
                points = int(values[0])
            elif key == "DATA":
                if values[0] != "binary":
                    raise ValueError(f"Only binary PCD is supported, got {values[0]} in {path}")
                break

        if fields is None or sizes is None or types is None:
            raise ValueError(f"Incomplete PCD header in {path}")
        if counts is None:
            counts = [1] * len(fields)
        if points is None:
            raise ValueError(f"Missing POINTS count in {path}")

        dtype_fields = []
        for field, size, pcd_type, count in zip(fields, sizes, types, counts):
            if count != 1:
                raise ValueError(f"Unsupported PCD COUNT {count} for field {field} in {path}")
            if pcd_type == "F" and size == 4:
                dtype = "<f4"
            elif pcd_type == "F" and size == 8:
                dtype = "<f8"
            elif pcd_type == "I":
                dtype = f"<i{size}"
            elif pcd_type == "U":
                dtype = f"<u{size}"
            else:
                raise ValueError(f"Unsupported PCD field type {pcd_type}{size} in {path}")
            dtype_fields.append((field, dtype))

        payload = f.read()

    data = np.frombuffer(payload, dtype=np.dtype(dtype_fields), count=points)
    return np.column_stack((data["x"], data["y"], data["z"])).astype(np.float64, copy=False)


def count_points_in_bbox(points: np.ndarray, bbox: list[float]) -> int:
    x, y, z, dx, dy, dz, yaw = [float(v) for v in bbox]
    if len(points) == 0:
        return 0

    rel = points - np.array([x, y, z], dtype=np.float64)
    c = math.cos(yaw)
    s = math.sin(yaw)

    local_x = c * rel[:, 0] + s * rel[:, 1]
    local_y = -s * rel[:, 0] + c * rel[:, 1]
    local_z = rel[:, 2]

    eps = 1e-9
    mask = (
        (np.abs(local_x) <= dx / 2.0 + eps)
        & (np.abs(local_y) <= dy / 2.0 + eps)
        & (np.abs(local_z) <= dz / 2.0 + eps)
    )
    return int(np.count_nonzero(mask))


def transfer_labels_for_sensor(
    experiment_dir: Path,
    sensor: str,
    source_label_dir: str,
    target_label_dir: str,
    overwrite: bool,
    drop_empty: bool,
) -> tuple[int, int, int]:
    source_dir = experiment_dir / source_label_dir
    target_dir = experiment_dir / target_label_dir
    pcd_dir = experiment_dir / f"{sensor}_pcd_transform"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source label directory: {source_dir}")
    if not pcd_dir.is_dir():
        raise FileNotFoundError(f"Missing transformed PCD directory: {pcd_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_existing = 0
    dropped_empty = 0

    for label_path in sorted(source_dir.glob("*.json")):
        target_path = target_dir / label_path.name
        if target_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        pcd_path = pcd_dir / f"{label_path.stem}.pcd"
        if not pcd_path.exists():
            raise FileNotFoundError(f"Missing PCD for label {label_path}: {pcd_path}")

        points = read_pcd_xyz(pcd_path)
        with label_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)

        if is_ignore_frame_annotation(labels):
            with target_path.open("w", encoding="utf-8") as f:
                json.dump({IGNORE_FRAME_KEY: True}, f, indent=2)
            written += 1
            continue

        if not isinstance(labels, list):
            raise ValueError(f"Expected a label list in {label_path}")

        transferred = []
        for row in labels:
            if "bbox" not in row or "label" not in row:
                raise ValueError(f"Missing label/bbox in {label_path}")

            updated = dict(row)
            num_points = count_points_in_bbox(points, updated["bbox"])
            if drop_empty and num_points == 0:
                dropped_empty += 1
                continue
            updated["bbox"] = [float(v) for v in updated["bbox"]]
            updated["num_lidar_pts"] = num_points
            transferred.append(updated)

        with target_path.open("w", encoding="utf-8") as f:
            json.dump(transferred, f, indent=2)
        written += 1

    return written, skipped_existing, dropped_empty


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy manual merged-view labels to os0/os1 manual label folders "
            "and recompute num_lidar_pts in each transformed sensor point cloud."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiment folder names. Defaults to *_experiment_car_*.",
    )
    parser.add_argument("--source-label-dir", default="merged_labels_manual")
    parser.add_argument("--os0-label-dir", default="os0_labels_manual")
    parser.add_argument("--os1-label-dir", default="os1_labels_manual")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop boxes with zero points in the target sensor point cloud.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if args.experiments:
        experiments = [data_root / name for name in args.experiments]
    else:
        experiments = sorted(data_root.glob("*_experiment_car_*"))

    if not experiments:
        raise FileNotFoundError(f"No car experiments found in {data_root}")

    for experiment_dir in experiments:
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"Missing experiment directory: {experiment_dir}")

        print(f"{experiment_dir.name}:")
        os0_stats = transfer_labels_for_sensor(
            experiment_dir,
            "os0",
            args.source_label_dir,
            args.os0_label_dir,
            args.overwrite,
            args.drop_empty,
        )
        os1_stats = transfer_labels_for_sensor(
            experiment_dir,
            "os1",
            args.source_label_dir,
            args.os1_label_dir,
            args.overwrite,
            args.drop_empty,
        )
        print(
            f"  os0: written={os0_stats[0]} skipped={os0_stats[1]} "
            f"dropped_empty={os0_stats[2]}"
        )
        print(
            f"  os1: written={os1_stats[0]} skipped={os1_stats[1]} "
            f"dropped_empty={os1_stats[2]}"
        )


if __name__ == "__main__":
    main()
