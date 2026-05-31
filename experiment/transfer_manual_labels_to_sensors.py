import argparse
import bisect
import json
import math
import sqlite3
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
INVALID_FRAME_KEY = "invalid_frame"


def is_invalid_frame_annotation(data: object) -> bool:
    if isinstance(data, dict):
        return bool(data.get(INVALID_FRAME_KEY, False) or data.get(IGNORE_FRAME_KEY, False))
    if isinstance(data, list):
        return any(
            bool(item.get(INVALID_FRAME_KEY, False) or item.get(IGNORE_FRAME_KEY, False))
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


def score_points_in_bbox(points: np.ndarray, bbox: list[float], z_min_margin: float = 0.0) -> int:
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
    z_min = -dz / 2.0 + float(z_min_margin)
    mask = (
        (np.abs(local_x) <= dx / 2.0 + eps)
        & (np.abs(local_y) <= dy / 2.0 + eps)
        & (local_z >= z_min - eps)
        & (local_z <= dz / 2.0 + eps)
    )
    return int(np.count_nonzero(mask))


def shift_bbox_local_xy(bbox: list[float], local_dx: float, local_dy: float) -> list[float]:
    shifted = [float(v) for v in bbox]
    yaw = shifted[6]
    c = math.cos(yaw)
    s = math.sin(yaw)
    shifted[0] += c * local_dx - s * local_dy
    shifted[1] += s * local_dx + c * local_dy
    return shifted


def refine_bbox_xy_to_points(
    points: np.ndarray,
    bbox: list[float],
    fit_range: float,
    fit_step: float,
    fit_z_min_margin: float,
) -> tuple[list[float], int, int, tuple[float, float]]:
    if fit_range <= 0.0 or fit_step <= 0.0:
        base_score = score_points_in_bbox(points, bbox, fit_z_min_margin)
        return [float(v) for v in bbox], base_score, base_score, (0.0, 0.0)

    offsets = np.arange(-fit_range, fit_range + fit_step * 0.5, fit_step)
    base_score = score_points_in_bbox(points, bbox, fit_z_min_margin)
    best_bbox = [float(v) for v in bbox]
    best_score = base_score
    best_shift = (0.0, 0.0)
    best_shift_norm = 0.0

    for local_dx in offsets:
        for local_dy in offsets:
            local_dx = float(local_dx)
            local_dy = float(local_dy)
            shift_norm = local_dx * local_dx + local_dy * local_dy
            candidate = shift_bbox_local_xy(bbox, local_dx, local_dy)
            score = score_points_in_bbox(points, candidate, fit_z_min_margin)
            if score > best_score or (score == best_score and shift_norm < best_shift_norm):
                best_bbox = candidate
                best_score = score
                best_shift = (local_dx, local_dy)
                best_shift_norm = shift_norm

    return best_bbox, best_score, base_score, best_shift


def read_point_timestamps_from_bag(experiment_dir: Path, sensor: str) -> list[int]:
    bag_dirs = sorted(experiment_dir.glob(f"{sensor}_rosbag2_*"))
    if not bag_dirs:
        raise FileNotFoundError(f"Missing {sensor} ROS bag directory in {experiment_dir}")
    db_paths = sorted(bag_dirs[0].glob("*.db3"))
    if not db_paths:
        raise FileNotFoundError(f"Missing db3 file in {bag_dirs[0]}")

    con = sqlite3.connect(db_paths[0])
    cur = con.cursor()
    topics = cur.execute("select id, name from topics").fetchall()
    point_topic_ids = [
        topic_id for topic_id, name in topics
        if name == f"/ouster_{sensor}/points"
    ]
    if not point_topic_ids:
        con.close()
        raise ValueError(f"Could not find /ouster_{sensor}/points in {db_paths[0]}")

    timestamps = [
        row[0]
        for row in cur.execute(
            "select timestamp from messages where topic_id=? order by timestamp",
            (point_topic_ids[0],),
        )
    ]
    con.close()
    return timestamps


def closest_timestamp(source: int, candidates: list[int]) -> int:
    idx = bisect.bisect_left(candidates, source)
    closest = []
    if idx < len(candidates):
        closest.append(candidates[idx])
    if idx > 0:
        closest.append(candidates[idx - 1])
    if not closest:
        raise ValueError("No timestamps available for matching")
    return min(closest, key=lambda candidate: abs(candidate - source))


def build_os1_label_trajectory(
    experiment_dir: Path,
    source_label_dir: str,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    os0_timestamps = read_point_timestamps_from_bag(experiment_dir, "os0")
    os1_timestamps = read_point_timestamps_from_bag(experiment_dir, "os1")
    os0_set = set(os0_timestamps)
    source_dir = experiment_dir / source_label_dir

    trajectory_times = []
    trajectory_xy = []
    offsets_ms = []

    for label_path in sorted(source_dir.glob("*.json")):
        os0_time = int(label_path.stem)
        if os0_time not in os0_set:
            continue
        with label_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        if is_invalid_frame_annotation(labels) or not isinstance(labels, list):
            continue
        rows = [row for row in labels if isinstance(row, dict) and "bbox" in row]
        if not rows:
            continue

        os1_time = closest_timestamp(os0_time, os1_timestamps)
        bbox = rows[0]["bbox"]
        trajectory_times.append(os1_time)
        trajectory_xy.append([float(bbox[0]), float(bbox[1])])
        offsets_ms.append((os1_time - os0_time) / 1e6)

    if len(trajectory_times) < 2:
        raise ValueError(
            f"Need at least two valid labels to build time correction for {experiment_dir}"
        )

    order = np.argsort(np.asarray(trajectory_times, dtype=np.int64))
    times = np.asarray(trajectory_times, dtype=np.float64)[order]
    xy = np.asarray(trajectory_xy, dtype=np.float64)[order]
    return times, xy, offsets_ms


def interp_with_extrapolation(times: np.ndarray, values: np.ndarray, target: float) -> float:
    if target < times[0]:
        t0, t1 = times[0], times[1]
        v0, v1 = values[0], values[1]
        return float(v0 + (target - t0) * (v1 - v0) / (t1 - t0))
    if target > times[-1]:
        t0, t1 = times[-2], times[-1]
        v0, v1 = values[-2], values[-1]
        return float(v1 + (target - t1) * (v1 - v0) / (t1 - t0))
    return float(np.interp(target, times, values))


def build_os0_time_correction(
    experiment_dir: Path,
    source_label_dir: str,
    extra_offset_ms: float = 0.0,
):
    times, xy, offsets_ms = build_os1_label_trajectory(experiment_dir, source_label_dir)
    extra_offset_ns = float(extra_offset_ms) * 1e6

    def correct_bbox_to_os0_time(bbox: list[float], frame_id: str) -> list[float]:
        target_time = float(int(frame_id)) + extra_offset_ns
        corrected = [float(v) for v in bbox]
        corrected[0] = interp_with_extrapolation(times, xy[:, 0], target_time)
        corrected[1] = interp_with_extrapolation(times, xy[:, 1], target_time)
        return corrected

    return correct_bbox_to_os0_time, offsets_ms


def estimate_os0_time_offset_ms(
    experiment_dir: Path,
    source_label_dir: str,
    fit_time_range_ms: float,
    fit_time_step_ms: float,
    fit_time_sample_limit: int,
    fit_z_min_margin: float,
) -> tuple[float, int, int]:
    times, xy, _ = build_os1_label_trajectory(experiment_dir, source_label_dir)
    source_dir = experiment_dir / source_label_dir
    pcd_dir = experiment_dir / "os0_pcd_transform"
    label_paths = []

    for label_path in sorted(source_dir.glob("*.json")):
        with label_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        if is_invalid_frame_annotation(labels) or not isinstance(labels, list):
            continue
        if any(isinstance(row, dict) and "bbox" in row for row in labels):
            label_paths.append(label_path)

    if fit_time_sample_limit > 0 and len(label_paths) > fit_time_sample_limit:
        sample_indices = np.linspace(0, len(label_paths) - 1, fit_time_sample_limit).round()
        label_paths = [label_paths[int(idx)] for idx in sample_indices]

    candidates = np.arange(
        -fit_time_range_ms,
        fit_time_range_ms + fit_time_step_ms * 0.5,
        fit_time_step_ms,
    )
    if len(candidates) == 0:
        raise ValueError("No os0 time-offset candidates to evaluate")

    scores = np.zeros(len(candidates), dtype=np.float64)
    evaluated_boxes = 0

    for label_path in label_paths:
        pcd_path = pcd_dir / f"{label_path.stem}.pcd"
        if not pcd_path.exists():
            raise FileNotFoundError(f"Missing PCD for label {label_path}: {pcd_path}")
        points = read_pcd_xyz(pcd_path)
        frame_time = float(int(label_path.stem))
        with label_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)

        for row in labels:
            if not isinstance(row, dict) or "bbox" not in row:
                continue
            bbox = [float(v) for v in row["bbox"]]
            evaluated_boxes += 1
            for idx, candidate_ms in enumerate(candidates):
                target_time = frame_time + float(candidate_ms) * 1e6
                candidate_bbox = list(bbox)
                candidate_bbox[0] = interp_with_extrapolation(times, xy[:, 0], target_time)
                candidate_bbox[1] = interp_with_extrapolation(times, xy[:, 1], target_time)
                scores[idx] += score_points_in_bbox(
                    points,
                    candidate_bbox,
                    fit_z_min_margin,
                )

    best_score = float(np.max(scores))
    best_indices = np.flatnonzero(scores == best_score)
    best_idx = min(best_indices, key=lambda idx: abs(float(candidates[idx])))
    return float(candidates[best_idx]), int(best_score), evaluated_boxes


def transfer_labels_for_sensor(
    experiment_dir: Path,
    sensor: str,
    source_label_dir: str,
    target_label_dir: str,
    overwrite: bool,
    drop_empty: bool,
    bbox_correction=None,
    fit_to_points: bool = False,
    fit_range: float = 0.35,
    fit_step: float = 0.05,
    fit_z_min_margin: float = 0.15,
) -> tuple[int, int, int, list[tuple[float, float]], list[int]]:
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
    fit_shifts = []
    fit_improvements = []

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

        if is_invalid_frame_annotation(labels):
            with target_path.open("w", encoding="utf-8") as f:
                json.dump({INVALID_FRAME_KEY: True}, f, indent=2)
            written += 1
            continue

        if not isinstance(labels, list):
            raise ValueError(f"Expected a label list in {label_path}")

        transferred = []
        for row in labels:
            if "bbox" not in row or "label" not in row:
                raise ValueError(f"Missing label/bbox in {label_path}")

            updated = dict(row)
            updated["bbox"] = [float(v) for v in updated["bbox"]]
            if bbox_correction is not None:
                updated["bbox"] = bbox_correction(updated["bbox"], label_path.stem)
            if fit_to_points:
                fitted_bbox, fitted_score, base_score, shift = refine_bbox_xy_to_points(
                    points,
                    updated["bbox"],
                    fit_range,
                    fit_step,
                    fit_z_min_margin,
                )
                updated["bbox"] = fitted_bbox
                fit_shifts.append(shift)
                fit_improvements.append(fitted_score - base_score)
            num_points = count_points_in_bbox(points, updated["bbox"])
            if drop_empty and num_points == 0:
                dropped_empty += 1
                continue
            updated["num_lidar_pts"] = num_points
            transferred.append(updated)

        with target_path.open("w", encoding="utf-8") as f:
            json.dump(transferred, f, indent=2)
        written += 1

    return written, skipped_existing, dropped_empty, fit_shifts, fit_improvements


def format_fit_stats(fit_shifts: list[tuple[float, float]], fit_improvements: list[int]) -> str:
    if not fit_shifts:
        return "fit_boxes=0"
    shifts = np.asarray(fit_shifts, dtype=np.float64)
    shift_norms = np.linalg.norm(shifts, axis=1)
    improvements = np.asarray(fit_improvements, dtype=np.int64)
    return (
        f"fit_boxes={len(fit_shifts)} "
        f"median_local_shift=({float(np.median(shifts[:, 0])):.3f},"
        f"{float(np.median(shifts[:, 1])):.3f})m "
        f"mean_shift_m={float(np.mean(shift_norms)):.3f} "
        f"max_shift_m={float(np.max(shift_norms)):.3f} "
        f"mean_score_gain={float(np.mean(improvements)):.1f}"
    )


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
    parser.add_argument(
        "--time-correct-os0",
        action="store_true",
        help=(
            "Treat merged labels as aligned to the matched os1 frame and "
            "interpolate bbox centers back to the true os0 timestamp."
        ),
    )
    parser.add_argument(
        "--auto-fit-os0-time-offset",
        action="store_true",
        help=(
            "Estimate an additional global os0 time offset by scanning for the "
            "trajectory position with the best os0 point fit."
        ),
    )
    parser.add_argument(
        "--fit-time-range-ms",
        type=float,
        default=80.0,
        help="Time-offset search radius in milliseconds for --auto-fit-os0-time-offset.",
    )
    parser.add_argument(
        "--fit-time-step-ms",
        type=float,
        default=5.0,
        help="Time-offset search step in milliseconds for --auto-fit-os0-time-offset.",
    )
    parser.add_argument(
        "--fit-time-sample-limit",
        type=int,
        default=50,
        help="Maximum valid frames used for time-offset fitting; 0 uses all frames.",
    )
    parser.add_argument(
        "--fit-os0-to-points",
        action="store_true",
        help=(
            "After optional time correction, locally adjust os0 bbox x/y centers "
            "to maximize points inside the box. Dimensions, height and yaw stay fixed."
        ),
    )
    parser.add_argument(
        "--fit-range",
        type=float,
        default=0.35,
        help="Local x/y search radius in meters for --fit-os0-to-points.",
    )
    parser.add_argument(
        "--fit-step",
        type=float,
        default=0.05,
        help="Local x/y search step in meters for --fit-os0-to-points.",
    )
    parser.add_argument(
        "--fit-z-min-margin",
        type=float,
        default=0.15,
        help=(
            "Ignore the bottom part of the box for the point-fit score, in meters, "
            "to reduce ground influence."
        ),
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
        os0_bbox_correction = None
        os0_extra_offset_ms = 0.0
        if args.auto_fit_os0_time_offset:
            os0_extra_offset_ms, best_score, evaluated_boxes = estimate_os0_time_offset_ms(
                experiment_dir,
                args.source_label_dir,
                args.fit_time_range_ms,
                args.fit_time_step_ms,
                args.fit_time_sample_limit,
                args.fit_z_min_margin,
            )
            print(
                "  os0 auto time-offset fit: "
                f"extra_offset_ms={os0_extra_offset_ms:.3f} "
                f"score={best_score} boxes={evaluated_boxes}"
            )
        if args.time_correct_os0 or args.auto_fit_os0_time_offset:
            os0_bbox_correction, offsets_ms = build_os0_time_correction(
                experiment_dir,
                args.source_label_dir,
                extra_offset_ms=os0_extra_offset_ms,
            )
            print(
                "  os0 time correction from os1 trajectory: "
                f"median_offset_ms={float(np.median(offsets_ms)):.3f} "
                f"mean_offset_ms={float(np.mean(offsets_ms)):.3f} "
                f"extra_offset_ms={os0_extra_offset_ms:.3f}"
            )
        os0_stats = transfer_labels_for_sensor(
            experiment_dir,
            "os0",
            args.source_label_dir,
            args.os0_label_dir,
            args.overwrite,
            args.drop_empty,
            bbox_correction=os0_bbox_correction,
            fit_to_points=args.fit_os0_to_points,
            fit_range=args.fit_range,
            fit_step=args.fit_step,
            fit_z_min_margin=args.fit_z_min_margin,
        )
        os1_stats = transfer_labels_for_sensor(
            experiment_dir,
            "os1",
            args.source_label_dir,
            args.os1_label_dir,
            args.overwrite,
            args.drop_empty,
            bbox_correction=None,
        )
        print(
            f"  os0: written={os0_stats[0]} skipped={os0_stats[1]} "
            f"dropped_empty={os0_stats[2]}"
        )
        if args.fit_os0_to_points:
            print(f"    os0 point fit: {format_fit_stats(os0_stats[3], os0_stats[4])}")
        print(
            f"  os1: written={os1_stats[0]} skipped={os1_stats[1]} "
            f"dropped_empty={os1_stats[2]}"
        )


if __name__ == "__main__":
    main()
