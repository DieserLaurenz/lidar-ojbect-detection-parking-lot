import argparse
import json
from pathlib import Path

from pcd_annotate import STATIC_LABELS
from transfer_manual_labels_to_sensors import (
    INVALID_FRAME_KEY,
    count_points_in_bbox,
    is_invalid_frame_annotation,
    read_pcd_xyz,
)


DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "2025_10_09"
    / "Experiment-Data"
    / "experiments"
)

SENSOR_PCD_DIRS = {
    "merged": "merged_pcd",
    "os0": "os0_pcd_transform",
    "os1": "os1_pcd_transform",
}


def static_label_applies_to_sensor(static_label: dict, sensor: str) -> bool:
    if sensor == "merged":
        return True
    return sensor in static_label.get("sensors", [])


def bbox_already_present(labels: list[dict], bbox: list[float], eps: float = 1e-6) -> bool:
    for label in labels:
        existing = label.get("bbox")
        if not isinstance(existing, list) or len(existing) != len(bbox):
            continue
        if all(abs(float(a) - float(b)) <= eps for a, b in zip(existing, bbox)):
            return True
    return False


def add_static_labels_for_sensor(
    experiment_dir: Path,
    sensor: str,
    source_label_dir: str,
    target_label_dir: str,
    threshold_ratio: float,
    min_static_points: int,
    overwrite: bool,
) -> tuple[int, int, int, int]:
    source_dir = experiment_dir / source_label_dir
    target_dir = experiment_dir / target_label_dir
    pcd_dir = experiment_dir / SENSOR_PCD_DIRS[sensor]

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source label directory: {source_dir}")
    if not pcd_dir.is_dir():
        raise FileNotFoundError(f"Missing PCD directory: {pcd_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_existing = 0
    invalid_frames = 0
    added_static = 0

    for label_path in sorted(source_dir.glob("*.json")):
        target_path = target_dir / label_path.name
        if target_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        with label_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)

        if is_invalid_frame_annotation(labels):
            with target_path.open("w", encoding="utf-8") as f:
                json.dump({INVALID_FRAME_KEY: True}, f, indent=2)
            written += 1
            invalid_frames += 1
            continue

        if not isinstance(labels, list):
            raise ValueError(f"Expected label list in {label_path}")

        pcd_path = pcd_dir / f"{label_path.stem}.pcd"
        if not pcd_path.exists():
            raise FileNotFoundError(f"Missing PCD for label {label_path}: {pcd_path}")
        points = read_pcd_xyz(pcd_path)

        merged_labels = [dict(label) for label in labels]
        for static_label in STATIC_LABELS:
            if static_label.get("class") != "car":
                continue
            if not static_label_applies_to_sensor(static_label, sensor):
                continue

            bbox = [float(value) for value in static_label["bbox"]]
            if bbox_already_present(merged_labels, bbox):
                continue

            num_points = count_points_in_bbox(points, bbox)
            reference_threshold = float(static_label["num_pts"]) * threshold_ratio
            required_points = min(reference_threshold, float(min_static_points))
            if num_points < required_points:
                continue

            merged_labels.append(
                {
                    "label": "car",
                    "bbox": bbox,
                    "num_lidar_pts": num_points,
                    "static": True,
                }
            )
            added_static += 1

        with target_path.open("w", encoding="utf-8") as f:
            json.dump(merged_labels, f, indent=2)
        written += 1

    return written, skipped_existing, invalid_frames, added_static


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy manual car labels into separate folders and add hard-coded "
            "static parked-car labels when enough points are present."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiment folder names. Defaults to *_experiment_car_*.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.5,
        help=(
            "Reference threshold ratio from the original static labels. The effective "
            "threshold is capped by --min-static-points because the reference counts "
            "come from a different sensor/view density."
        ),
    )
    parser.add_argument(
        "--min-static-points",
        type=int,
        default=120,
        help="Absolute minimum points required inside a static car box.",
    )
    parser.add_argument(
        "--merged-source-label-dir",
        default="merged_labels_manual",
    )
    parser.add_argument(
        "--os0-source-label-dir",
        default="os0_labels_manual",
    )
    parser.add_argument(
        "--os1-source-label-dir",
        default="os1_labels_manual",
    )
    parser.add_argument(
        "--target-suffix",
        default="_static",
        help="Suffix appended to each source label directory.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if args.experiments:
        experiments = [data_root / name for name in args.experiments]
    else:
        experiments = sorted(data_root.glob("*_experiment_car_*"))

    if not experiments:
        raise FileNotFoundError(f"No car experiments found in {data_root}")

    source_dirs = {
        "merged": args.merged_source_label_dir,
        "os0": args.os0_source_label_dir,
        "os1": args.os1_source_label_dir,
    }

    for experiment_dir in experiments:
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"Missing experiment directory: {experiment_dir}")
        print(f"{experiment_dir.name}:")

        for sensor, source_label_dir in source_dirs.items():
            target_label_dir = f"{source_label_dir}{args.target_suffix}"
            written, skipped, invalid, added = add_static_labels_for_sensor(
                experiment_dir,
                sensor,
                source_label_dir,
                target_label_dir,
                args.threshold_ratio,
                args.min_static_points,
                args.overwrite,
            )
            print(
                f"  {sensor}: written={written} skipped={skipped} "
                f"invalid={invalid} added_static={added} target={target_label_dir}"
            )


if __name__ == "__main__":
    main()
