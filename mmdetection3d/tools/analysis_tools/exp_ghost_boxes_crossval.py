"""Count class-wise and total OOF false-positive boxes for the CV models.

A "ghost box" is defined here as a prediction with score >= 0.3 that cannot
be greedily matched to an as-yet-unmatched GT box of the same class at 3D IoU
>= 0.3.  This generalizes the existing bicycle-FP diagnostic to all three
classes without changing training or the official AP evaluation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from exp_eval_crossval import CLASSES, FOLDS, VIEWS, iou_matrix, latest


def count_class(
    predictions: list,
    samples: list[dict],
    class_index: int,
    device: str,
    score_threshold: float,
    iou_threshold: float,
) -> dict:
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    totals = {
        "frames": len(samples),
        "considered_predictions": 0,
        "true_positives": 0,
        "false_positives": 0,
        "frames_with_false_positive": 0,
    }
    fp_frames = set()

    for result in predictions:
        sample_id = str(result.metainfo.get("sample_id", ""))
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue

        pred_boxes = result.bboxes_3d.tensor.numpy()[:, :7]
        pred_labels = np.asarray(result.labels_3d)
        pred_scores = np.asarray(result.scores_3d)
        mask = (pred_labels == class_index) & (pred_scores >= score_threshold)
        boxes = pred_boxes[mask]
        scores = pred_scores[mask]
        boxes = boxes[np.argsort(-scores)]
        gt_boxes = np.asarray([
            instance["bbox_3d"][:7]
            for instance in sample.get("instances", [])
            if int(instance["bbox_label_3d"]) == class_index
        ], dtype=np.float64).reshape(-1, 7)

        ious = iou_matrix(boxes, gt_boxes, device)
        matched = [False] * len(gt_boxes)
        totals["considered_predictions"] += len(boxes)
        for pred_index in range(len(boxes)):
            if len(gt_boxes):
                match_index = int(np.argmax(ious[pred_index]))
                match_iou = float(ious[pred_index, match_index])
            else:
                match_index = -1
                match_iou = 0.0
            if match_iou >= iou_threshold and not matched[match_index]:
                matched[match_index] = True
                totals["true_positives"] += 1
            else:
                totals["false_positives"] += 1
                fp_frames.add(sample_id)

    totals["frames_with_false_positive"] = len(fp_frames)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/exp"))
    parser.add_argument("--results-root", type=Path,
                        default=Path("results/crossval"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--output", type=Path,
                        default=Path("results/crossval/GHOST_BOXES.json"))
    args = parser.parse_args()

    report = {
        "definition": (
            "score >= threshold and no greedy same-class GT match at "
            "3D IoU >= threshold"
        ),
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "views": {},
    }
    for view in VIEWS:
        pooled = {
            name: {"frames": 0, "considered_predictions": 0,
                   "true_positives": 0, "false_positives": 0,
                   "frames_with_false_positive": 0}
            for name in CLASSES
        }
        for fold in FOLDS:
            prediction_path = latest(str(
                args.results_root /
                f"{view}_cv{fold}_gtsample_test_results_*.pkl"
            ))
            info_path = (
                args.data_root / f"exp_kitti_infos_{view}_cv{fold}_test.pkl"
            )
            with prediction_path.open("rb") as handle:
                predictions = pickle.load(handle)
            with info_path.open("rb") as handle:
                samples = pickle.load(handle)["data_list"]
            for class_index, class_name in enumerate(CLASSES):
                values = count_class(
                    predictions, samples, class_index, args.device,
                    args.score_threshold, args.iou_threshold,
                )
                for key, value in values.items():
                    pooled[class_name][key] += value

        frames = next(iter(pooled.values()))["frames"]
        total_fp = sum(values["false_positives"] for values in pooled.values())
        report["views"][view] = {
            "frames": frames,
            "classes": pooled,
            "total_false_positives": total_fp,
            "total_false_positives_per_100_frames": 100.0 * total_fp / frames,
        }
        print(view, {
            name: values["false_positives"]
            for name, values in pooled.items()
        }, "total", total_fp, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
