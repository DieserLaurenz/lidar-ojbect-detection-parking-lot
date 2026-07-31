"""Evaluate experiment-held-out cross-validation predictions.

The script reproduces the dataset-level AP11 metric for every fold, pools the
out-of-fold predictions, and repeats the evaluation for dynamic/static GT.
Non-target GT acts as an ignore region in the split evaluations, matching
``exp_eval_dynamic_static.py``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mmcv.ops import diff_iou_rotated_3d


VIEWS = ("merged", "os0", "os1")
FOLDS = (1, 2, 3)
THRESHOLDS = (0.3, 0.4, 0.5, 0.6)
TARGETS = ("all", "dynamic", "static")
CLASSES = ("person", "bicycle", "car")
TYPE_NUM = {"merged": 2, "os0": 0, "os1": 1}
LUMPI_IDX = {"person": 0, "car": 1, "bicycle": 2, "bike": 2}
KITTI_MAP = {0: 0, 1: 2, 2: 1}

THETA = np.deg2rad(90)
R = np.array(
    [[np.cos(THETA), -np.sin(THETA), 0],
     [np.sin(THETA), np.cos(THETA), 0],
     [0, 0, 1]],
    dtype=np.float64,
)
T = np.array([30.0, 0.0, -1.6], dtype=np.float64)


def latest(pattern: str) -> Path:
    matches = [Path(path) for path in glob.glob(pattern)]
    if not matches:
        raise FileNotFoundError(pattern)
    return max(matches, key=lambda path: path.stat().st_mtime)


def build_static_lookup(raw_root: Path, view: str) -> dict[str, list[tuple]]:
    """Map sample ID to (class, KITTI box, static flag) raw annotations."""
    lookup = {}
    for exp_dir in sorted(raw_root.glob("*_experiment_*")):
        measurement = exp_dir.name[0]
        raw_dir = None
        for candidate in (
            f"{view}_labels_manual_correct",
            f"{view}_labels_manual_static",
            f"{view}_labels_manual",
            f"{view}_labels",
        ):
            path = exp_dir / candidate
            if any(path.glob("*.json")):
                raw_dir = path
                break
        if raw_dir is None:
            continue

        for path in raw_dir.glob("*.json"):
            with path.open() as handle:
                data = json.load(handle)
            items = data if isinstance(data, list) else data.get("instances", [])
            frame = int(path.stem)
            sample_id = f"{measurement}{TYPE_NUM[view]}0000{frame:06d}"
            entries = []
            for item in items:
                if not isinstance(item, dict) or "bbox" not in item:
                    continue
                box = np.asarray(item["bbox"][:7], dtype=np.float64)
                kitti_box = box.copy()
                kitti_box[:3] = R @ box[:3] + T
                kitti_box[2] -= kitti_box[5] / 2.0
                kitti_box[6] = (
                    box[6] + THETA + np.pi
                ) % (2 * np.pi) - np.pi
                label = KITTI_MAP[LUMPI_IDX[item["label"]]]
                entries.append((label, kitti_box, bool(item.get("static", False))))
            lookup[sample_id] = entries
    return lookup


def iou_matrix(pred: np.ndarray, gt: np.ndarray, device: str) -> np.ndarray:
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    pred_tensor = torch.as_tensor(pred, dtype=torch.float32, device=device).clone()
    gt_tensor = torch.as_tensor(gt, dtype=torch.float32, device=device).clone()
    pred_tensor[:, 2] += pred_tensor[:, 5] / 2.0
    gt_tensor[:, 2] += gt_tensor[:, 5] / 2.0
    pred_expanded = pred_tensor.unsqueeze(1).expand(-1, len(gt), -1)
    gt_expanded = gt_tensor.unsqueeze(0).expand(len(pred), -1, -1)
    return diff_iou_rotated_3d(pred_expanded, gt_expanded).cpu().numpy()


def ap11(scores: list[float], tps: list[int], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, dtype=np.float64)[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    value = 0.0
    for level in np.linspace(0, 1, 11):
        mask = recall >= level - 1e-9
        value += precision[mask].max() if mask.any() else 0.0
    return value / 11.0


def empty_records() -> dict:
    return {
        target: {
            class_index: {
                threshold: {"scores": [], "tps": [], "n_gt": 0}
                for threshold in THRESHOLDS
            }
            for class_index in range(len(CLASSES))
        }
        for target in TARGETS
    }


def merge_records(destination: dict, source: dict) -> None:
    for target in TARGETS:
        for class_index in range(len(CLASSES)):
            for threshold in THRESHOLDS:
                dst = destination[target][class_index][threshold]
                src = source[target][class_index][threshold]
                dst["scores"].extend(src["scores"])
                dst["tps"].extend(src["tps"])
                dst["n_gt"] += src["n_gt"]


def static_flags_for_sample(
    sample: dict,
    raw_entries: list[tuple],
) -> tuple[list[tuple[int, np.ndarray]], list[bool], int]:
    gts = []
    flags = []
    unmatched = 0
    for instance in sample.get("instances", []):
        label = int(instance["bbox_label_3d"])
        box = np.asarray(instance["bbox_3d"][:7], dtype=np.float64)
        best_flag = None
        best_distance = math.inf
        for raw_label, raw_box, static in raw_entries:
            if raw_label != label:
                continue
            distance = float(np.linalg.norm(raw_box[:3] - box[:3]))
            if distance < best_distance:
                best_distance = distance
                best_flag = static
        if best_flag is None or best_distance > 0.1:
            unmatched += 1
            best_flag = False
        gts.append((label, box))
        flags.append(bool(best_flag))
    return gts, flags, unmatched


def accumulate(
    predictions: list,
    samples: list[dict],
    raw_lookup: dict,
    device: str,
    experiment: str | None = None,
    ignore_floor: float | None = None,
) -> tuple[dict, int]:
    records = empty_records()
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    unmatched = 0

    for result in predictions:
        sample_id = str(result.metainfo.get("sample_id", ""))
        if experiment is not None and sample_id[:1] != experiment:
            continue
        sample = samples_by_id.get(sample_id)
        raw_entries = raw_lookup.get(sample_id)
        if sample is None or raw_entries is None:
            continue

        pred_boxes = result.bboxes_3d.tensor.numpy()[:, :7]
        pred_labels = np.asarray(result.labels_3d)
        pred_scores = np.asarray(result.scores_3d)
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]
        pred_scores = pred_scores[order]

        gts, flags, missing = static_flags_for_sample(sample, raw_entries)
        unmatched += missing
        for class_index in range(len(CLASSES)):
            gt_indices = [
                index for index, (label, _) in enumerate(gts)
                if label == class_index
            ]
            gt_boxes = np.asarray(
                [gts[index][1] for index in gt_indices], dtype=np.float64
            ).reshape(-1, 7)
            gt_static = [flags[index] for index in gt_indices]
            pred_indices = np.where(pred_labels == class_index)[0]
            class_pred_boxes = pred_boxes[pred_indices]
            class_pred_scores = pred_scores[pred_indices]
            ious = iou_matrix(class_pred_boxes, gt_boxes, device)

            target_masks = {
                "all": [True] * len(gt_indices),
                "dynamic": [not value for value in gt_static],
                "static": list(gt_static),
            }
            for target, target_mask in target_masks.items():
                for threshold in THRESHOLDS:
                    record = records[target][class_index][threshold]
                    record["n_gt"] += int(sum(target_mask))
                    matched = [False] * len(gt_indices)
                    for pred_index, score in enumerate(class_pred_scores):
                        if len(gt_indices):
                            match_index = int(np.argmax(ious[pred_index]))
                            match_iou = float(ious[pred_index, match_index])
                        else:
                            match_index = -1
                            match_iou = 0.0
                        if (
                            match_iou >= threshold
                            and not matched[match_index]
                        ):
                            matched[match_index] = True
                            if target_mask[match_index]:
                                record["scores"].append(float(score))
                                record["tps"].append(1)
                            # A match to non-target GT is ignored.
                        elif (
                            ignore_floor is not None
                            and match_index >= 0
                            and match_iou >= ignore_floor
                            and not target_mask[match_index]
                        ):
                            # Label-based ignore variant: the prediction lies
                            # on a non-target GT (e.g. a static-labeled car in
                            # the dynamic evaluation) without reaching the
                            # main threshold. The manual static flags make it
                            # attributable, so it is excluded from the
                            # ranking instead of counting as false positive.
                            pass
                        else:
                            record["scores"].append(float(score))
                            record["tps"].append(0)
    return records, unmatched


def metrics_from_records(records: dict) -> dict:
    output = {}
    ap_values = []
    for class_index, class_name in enumerate(CLASSES):
        output[f"n_{class_name}"] = records[class_index][THRESHOLDS[0]]["n_gt"]
        class_values = []
        for threshold in THRESHOLDS:
            record = records[class_index][threshold]
            value = ap11(record["scores"], record["tps"], record["n_gt"])
            output[f"AP{int(threshold * 100)}_{class_name}"] = value
            if not math.isnan(value):
                class_values.append(value)
                ap_values.append(value)
        output[f"mAP_{class_name}"] = (
            float(np.mean(class_values)) if class_values else float("nan")
        )
    output["mAP"] = float(np.mean(ap_values)) if ap_values else float("nan")
    return output


def bicycle_false_positives(
    predictions: list,
    samples: list[dict],
    device: str,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.3,
) -> dict:
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    frame_indices = {}
    grouped_ids = defaultdict(list)
    for sample_id in samples_by_id:
        grouped_ids[sample_id[0]].append(sample_id)
    for experiment, sample_ids in grouped_ids.items():
        sample_ids.sort(key=lambda value: int(value[6:]))
        for index, sample_id in enumerate(sample_ids):
            frame_indices[sample_id] = index

    false_positives = []
    true_positives = 0
    considered_predictions = 0
    for result in predictions:
        sample_id = str(result.metainfo.get("sample_id", ""))
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        pred_boxes = result.bboxes_3d.tensor.numpy()[:, :7]
        pred_labels = np.asarray(result.labels_3d)
        pred_scores = np.asarray(result.scores_3d)
        mask = (pred_labels == 1) & (pred_scores >= score_threshold)
        boxes = pred_boxes[mask]
        scores = pred_scores[mask]
        order = np.argsort(-scores)
        boxes = boxes[order]
        scores = scores[order]
        gt_boxes = np.asarray(
            [
                instance["bbox_3d"][:7]
                for instance in sample.get("instances", [])
                if int(instance["bbox_label_3d"]) == 1
            ],
            dtype=np.float64,
        ).reshape(-1, 7)
        ious = iou_matrix(boxes, gt_boxes, device)
        matched = [False] * len(gt_boxes)
        considered_predictions += len(boxes)
        for pred_index, (box, score) in enumerate(zip(boxes, scores)):
            if len(gt_boxes):
                match_index = int(np.argmax(ious[pred_index]))
                match_iou = float(ious[pred_index, match_index])
            else:
                match_index = -1
                match_iou = 0.0
            if match_iou >= iou_threshold and not matched[match_index]:
                matched[match_index] = True
                true_positives += 1
            else:
                false_positives.append({
                    "sample_id": sample_id,
                    "experiment": sample_id[0],
                    "frame_index": frame_indices[sample_id],
                    "score": float(score),
                    "center_xy": [float(box[0]), float(box[1])],
                })
    return {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "frames": len(samples),
        "considered_predictions": considered_predictions,
        "true_positives": true_positives,
        "false_positives": false_positives,
    }


def summarize_bicycle_false_positives(fold_data: list[dict]) -> dict:
    false_positives = [
        item
        for fold in fold_data
        for item in fold["false_positives"]
    ]
    frames = sum(fold["frames"] for fold in fold_data)
    considered = sum(fold["considered_predictions"] for fold in fold_data)
    true_positives = sum(fold["true_positives"] for fold in fold_data)
    scores = np.asarray([item["score"] for item in false_positives])

    isolated = 0
    for index, item in enumerate(false_positives):
        center = np.asarray(item["center_xy"])
        has_neighbor = False
        for other_index, other in enumerate(false_positives):
            if index == other_index or item["experiment"] != other["experiment"]:
                continue
            if abs(item["frame_index"] - other["frame_index"]) > 1:
                continue
            if np.linalg.norm(center - np.asarray(other["center_xy"])) <= 1.0:
                has_neighbor = True
                break
        if not has_neighbor:
            isolated += 1

    # Connected spatial components reveal repeated scene locations.
    centers = np.asarray([item["center_xy"] for item in false_positives])
    parent = list(range(len(false_positives)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            if np.linalg.norm(centers[left] - centers[right]) <= 1.0:
                union(left, right)
    components = defaultdict(list)
    for index in range(len(false_positives)):
        components[find(index)].append(index)
    clusters = []
    for indices in components.values():
        if len(indices) < 2:
            continue
        cluster_items = [false_positives[index] for index in indices]
        cluster_centers = np.asarray([item["center_xy"] for item in cluster_items])
        clusters.append({
            "count": len(indices),
            "center_xy": cluster_centers.mean(axis=0).tolist(),
            "experiments": sorted({item["experiment"] for item in cluster_items}),
            "median_score": float(np.median([item["score"] for item in cluster_items])),
        })
    clusters.sort(key=lambda cluster: cluster["count"], reverse=True)

    return {
        "score_threshold": 0.3,
        "iou_threshold": 0.3,
        "frames": frames,
        "considered_predictions": considered,
        "true_positives": true_positives,
        "false_positives": len(false_positives),
        "frames_with_false_positive": len({
            item["sample_id"] for item in false_positives
        }),
        "false_positives_per_100_frames": (
            100.0 * len(false_positives) / frames if frames else 0.0
        ),
        "isolated_one_frame_false_positives": isolated,
        "isolated_fraction": isolated / len(false_positives) if false_positives else 0.0,
        "score_median": float(np.median(scores)) if len(scores) else None,
        "score_p90": float(np.percentile(scores, 90)) if len(scores) else None,
        "score_max": float(np.max(scores)) if len(scores) else None,
        "spatial_clusters_ge_2": len(clusters),
        "largest_spatial_clusters": clusters[:10],
    }


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3f}"


def build_markdown(report: dict) -> str:
    lines = [
        "# Cross-Validation Results",
        "",
        "Experiment-held-out, paired-timestamp 3-fold evaluation. Values use "
        "dataset-level AP11 and the best validation checkpoint of each fold.",
        "",
        "## Aggregate performance",
        "",
        "| View | Fold 1 mAP | Fold 2 mAP | Fold 3 mAP | Mean +/- sample std | Pooled OOF mAP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for view in VIEWS:
        folds = [report["runs"][view][str(fold)]["all"]["mAP"] for fold in FOLDS]
        summary = report["fold_summary"][view]["all"]["mAP"]
        pooled = report["pooled"][view]["all"]["mAP"]
        lines.append(
            f"| {view} | {fmt(folds[0])} | {fmt(folds[1])} | {fmt(folds[2])} "
            f"| {fmt(summary['mean'])} +/- {fmt(summary['std'])} | {fmt(pooled)} |"
        )

    for target in ("dynamic", "static"):
        lines.extend([
            "",
            f"## {target.capitalize()} objects, pooled out-of-fold",
            "",
            "| View | person AP30 / AP60 | bicycle AP30 / AP60 | car AP30 / AP60 |",
            "|---|---:|---:|---:|",
        ])
        for view in VIEWS:
            metrics = report["pooled"][view][target]
            values = []
            for class_name in CLASSES:
                values.append(
                    f"{fmt(metrics[f'AP30_{class_name}'])} / "
                    f"{fmt(metrics[f'AP60_{class_name}'])} "
                    f"(n={metrics[f'n_{class_name}']})"
                )
            lines.append(f"| {view} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Dynamic target class by held-out experiment",
        "",
        "| Experiment | Target | merged AP30 / AP60 | os0 AP30 / AP60 | os1 AP30 / AP60 |",
        "|---:|---|---:|---:|---:|",
    ])
    target_class = {
        "1": "car", "2": "car", "3": "car",
        "4": "bicycle", "5": "bicycle", "6": "bicycle",
        "7": "person", "8": "person", "9": "person",
    }
    for experiment in map(str, range(1, 10)):
        class_name = target_class[experiment]
        cells = []
        for view in VIEWS:
            metrics = report["experiments"][view][experiment]["dynamic"]
            cells.append(
                f"{fmt(metrics[f'AP30_{class_name}'])} / "
                f"{fmt(metrics[f'AP60_{class_name}'])} "
                f"(n={metrics[f'n_{class_name}']})"
            )
        lines.append(
            f"| {experiment} | {class_name} | " + " | ".join(cells) + " |"
        )

    dynamic = {
        view: report["pooled"][view]["dynamic"] for view in VIEWS
    }
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `merged` has the highest aggregate mAP in every fold, but the mean "
        "advantage is modest (about 0.021 over os0 and 0.028 over os1). With "
        "only three folds, this supports a balanced-performance claim, not a "
        "strong significance claim.",
        f"- Dynamic car AP30/AP60 is merged {fmt(dynamic['merged']['AP30_car'])}/"
        f"{fmt(dynamic['merged']['AP60_car'])}, os0 "
        f"{fmt(dynamic['os0']['AP30_car'])}/{fmt(dynamic['os0']['AP60_car'])}, "
        f"and os1 {fmt(dynamic['os1']['AP30_car'])}/"
        f"{fmt(dynamic['os1']['AP60_car'])}. All views detect the moving car; "
        "os0 is slightly best. The old `os1=0.09 -> merged=0.90` headline was "
        "a temporal-split artifact.",
        "- Fusion's clearest benefits are person localization and bicycle "
        "localization. os0 has higher bicycle AP30 but substantially lower "
        "AP60, while merged is more precise spatially.",
        "- Dynamic car has exactly 339 paired GT instances in every view. "
        "Dynamic person/bicycle counts differ because annotations remain "
        "view-specific even though timestamps are paired; interpret those "
        "cross-view differences with that limitation.",
    ])

    lines.extend([
        "",
        "## Consistency",
        "",
        "The reproduced all-object fold mAP is compared with each metric JSON. "
        "The maximum absolute delta must remain below 1e-6.",
        "",
        f"Maximum reproduction delta: {report['max_official_mAP_delta']:.3e}",
        "",
        "Pooled AP combines predictions from differently trained fold models; "
        "therefore per-fold values and their variation remain primary. The "
        "pooled value is an additional out-of-fold summary, not a replacement "
        "for fold reporting.",
    ])
    lines.extend([
        "",
        "## Out-of-fold bicycle false positives",
        "",
        "Predictions use score >= 0.3 and are false positives when no unmatched "
        "bicycle GT reaches IoU 0.3 in the same frame.",
        "",
        "| View | FPs | FPs / 100 frames | Frames with FP | Isolated one-frame fraction | Median / p90 / max score |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for view in VIEWS:
        values = report["bicycle_false_positives"][view]
        lines.append(
            f"| {view} | {values['false_positives']} | "
            f"{values['false_positives_per_100_frames']:.2f} | "
            f"{values['frames_with_false_positive']} | "
            f"{values['isolated_fraction']:.1%} | "
            f"{fmt(values['score_median'])} / {fmt(values['score_p90'])} / "
            f"{fmt(values['score_max'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/exp"))
    parser.add_argument(
        "--raw-root", type=Path, default=Path.home() / "data" / "experiments"
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/crossval")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ignore-floor",
        type=float,
        default=None,
        help=(
            "Label-based ignore variant: exclude predictions overlapping a "
            "non-target GT with IoU >= FLOOR from the dynamic/static "
            "rankings (uses the manual static flags). Default None keeps "
            "the official metric unchanged; report variant runs under a "
            "separate output name."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/crossval/CROSS_VALIDATION_RESULTS.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("results/crossval/CROSS_VALIDATION_RESULTS.md"),
    )
    args = parser.parse_args()

    protocol = "paired experiment-held-out 3-fold cross-validation"
    if args.ignore_floor is not None:
        protocol += (
            f" (label-based ignore variant, floor {args.ignore_floor})"
        )
    report = {
        "protocol": protocol,
        "ignore_floor": args.ignore_floor,
        "thresholds": list(THRESHOLDS),
        "runs": {view: {} for view in VIEWS},
        "pooled": {},
        "fold_summary": {},
        "experiments": {view: {} for view in VIEWS},
        "bicycle_false_positives": {},
    }
    maximum_delta = 0.0

    for view in VIEWS:
        print(f"=== {view} ===", flush=True)
        raw_lookup = build_static_lookup(args.raw_root, view)
        pooled_records = empty_records()
        experiment_records = {}
        bicycle_fp_folds = []
        for fold in FOLDS:
            prediction_path = latest(
                str(args.results_root / f"{view}_cv{fold}_gtsample_test_results_*.pkl")
            )
            metric_path = latest(
                str(args.results_root / f"{view}_cv{fold}_gtsample_test_metric_*.json")
            )
            info_path = args.data_root / f"exp_kitti_infos_{view}_cv{fold}_test.pkl"
            with prediction_path.open("rb") as handle:
                predictions = pickle.load(handle)
            with info_path.open("rb") as handle:
                samples = pickle.load(handle)["data_list"]
            with metric_path.open() as handle:
                official = json.load(handle)["metric"]

            records, unmatched = accumulate(
                predictions, samples, raw_lookup, args.device,
                ignore_floor=args.ignore_floor,
            )
            merge_records(pooled_records, records)
            metrics = {
                target: metrics_from_records(records[target])
                for target in TARGETS
            }
            delta = abs(metrics["all"]["mAP"] - official["mAP"])
            maximum_delta = max(maximum_delta, delta)
            if delta > 1e-6:
                raise RuntimeError(
                    f"Metric reproduction failed for {view} fold {fold}: {delta}"
                )
            report["runs"][view][str(fold)] = {
                "prediction_file": str(prediction_path),
                "metric_file": str(metric_path),
                "info_file": str(info_path),
                "unmatched_static_flags": unmatched,
                **metrics,
            }
            fold_fp = bicycle_false_positives(
                predictions, samples, args.device
            )
            bicycle_fp_folds.append(fold_fp)
            report["runs"][view][str(fold)]["bicycle_false_positives"] = {
                key: value for key, value in fold_fp.items()
                if key != "false_positives"
            }
            report["runs"][view][str(fold)]["bicycle_false_positives"][
                "false_positives"
            ] = len(fold_fp["false_positives"])
            print(
                f"fold {fold}: mAP={metrics['all']['mAP']:.4f}, "
                f"dynamic car AP30={metrics['dynamic']['AP30_car']:.4f}, "
                f"unmatched={unmatched}",
                flush=True,
            )

            experiments = sorted({str(sample["sample_id"])[0] for sample in samples})
            for experiment in experiments:
                exp_records, exp_unmatched = accumulate(
                    predictions,
                    samples,
                    raw_lookup,
                    args.device,
                    experiment=experiment,
                    ignore_floor=args.ignore_floor,
                )
                experiment_records[experiment] = {
                    target: metrics_from_records(exp_records[target])
                    for target in TARGETS
                }
                experiment_records[experiment]["unmatched_static_flags"] = exp_unmatched

        report["pooled"][view] = {
            target: metrics_from_records(pooled_records[target])
            for target in TARGETS
        }
        report["experiments"][view] = experiment_records
        report["bicycle_false_positives"][view] = (
            summarize_bicycle_false_positives(bicycle_fp_folds)
        )
        report["fold_summary"][view] = {}
        for target in TARGETS:
            metric_names = report["runs"][view]["1"][target].keys()
            summary = {}
            for metric_name in metric_names:
                if metric_name.startswith("n_"):
                    continue
                values = [
                    report["runs"][view][str(fold)][target][metric_name]
                    for fold in FOLDS
                ]
                values = [value for value in values if not math.isnan(value)]
                summary[metric_name] = {
                    "mean": float(np.mean(values)) if values else float("nan"),
                    "std": (
                        float(np.std(values, ddof=1)) if len(values) > 1
                        else float("nan")
                    ),
                }
            report["fold_summary"][view][target] = summary

    report["max_official_mAP_delta"] = maximum_delta
    safe_report = json_safe(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(safe_report, indent=2) + "\n")
    args.output_markdown.write_text(build_markdown(report))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
