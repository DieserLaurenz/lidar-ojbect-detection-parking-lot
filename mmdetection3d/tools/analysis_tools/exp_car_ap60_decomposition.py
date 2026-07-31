"""Decompose the cross-validation car-AP60 gap of the merged view.

Answers why merged dynamic-car AP60 (0.739) trails os0 (0.847) and os1
(0.783) although merged has the highest aggregate mAP. For every view it
pools the out-of-fold predictions of the car experiments (fold f holds out
experiment f) and reports:

1. Box quality on the moving car: best-IoU distribution, dimension and
   center errors of the best-matching prediction per GT instance.
2. The precision side of the dynamic AP60 ranking: false positives
   categorized as sloppy boxes on static cars (IoU 0.25-0.6, no longer an
   ignorable match at the 0.6 threshold), sloppy boxes on the moving car,
   duplicates, and ghosts; plus high-confidence FP counts.
3. Static-car box quality incl. lidar-point statistics of the weakly boxed
   instances (identifies the sensor-adjacent parked car).

Run on the GPU server from the mmdetection3d repo root:
    python tools/analysis_tools/exp_car_ap60_decomposition.py

Finding (2026-07-12), pooled over all out-of-fold test frames (the exact
population of the official AP60; replication delta 0): merged boxes the
moving car most accurately of all views (median IoU 0.848, ~1 cm median
length error). Its AP60 deficit comes from one sensor-adjacent parked car
(x ~ 28.9, median ~18k points in merged, labeled 5.3 x 1.8 x 1.8 m) that
merged boxes at IoU 0.25-0.6 in nearly every frame (2,035 of 12,314 static
instances below IoU 0.6 vs 45/94 for os0/os1); at the 0.6 threshold those
predictions count as false positives of the dynamic evaluation (1,026 FPs
with score >= 0.8 vs 35/25 for os0/os1). Sensitivity: treating static
matches with IoU >= 0.25 as ignore lifts merged dynamic-car AP60 from
0.739 to 0.883 while os0/os1 move by <= 0.01. The same failure mode
explains merged static-car AP60 (0.814 vs ~0.91). See
results/CROSS_VALIDATION_RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from tools.analysis_tools.exp_eval_crossval import (  # noqa: E402
    ap11,
    build_static_lookup,
    iou_matrix,
    latest,
    static_flags_for_sample,
)

VIEWS = ("merged", "os0", "os1")
FOLDS = (1, 2, 3)
CAR = 2
IOU_MAIN = 0.6
IOU_SLOPPY = 0.25


def norm_dims(box: np.ndarray) -> tuple[float, float]:
    length, width = box[3], box[4]
    return (max(length, width), min(length, width))


def analyze_view(view: str, results_root: Path, data_root: Path,
                 raw_root: Path, device: str) -> dict:
    raw_lookup = build_static_lookup(raw_root, view)
    match_records = {"dynamic": [], "static": []}
    fp_categories = {"sloppy_static": 0, "sloppy_dyn": 0, "ghost": 0,
                     "dup": 0}
    fp_scores: list[float] = []
    # Ranking-Listen: offizielle Regel und Sensitivitätsvariante, bei der
    # Vorhersagen mit bestem Match auf statischem GT ab IoU 0.25 ebenfalls
    # ignoriert werden (statt als FP zu zählen).
    ranking = {"official": ([], []), "relaxed_ignore": ([], [])}
    weak_static_pts: list[float] = []
    good_static_pts: list[float] = []
    weak_static_x: list[float] = []
    tp_main = 0
    n_gt_dyn = 0

    for fold in FOLDS:
        prediction_path = latest(str(
            results_root / f"{view}_cv{fold}_gtsample_test_results_*.pkl"))
        with prediction_path.open("rb") as handle:
            predictions = pickle.load(handle)
        info_path = data_root / f"exp_kitti_infos_{view}_cv{fold}_test.pkl"
        with info_path.open("rb") as handle:
            samples = pickle.load(handle)["data_list"]
        samples_by_id = {str(s["sample_id"]): s for s in samples}

        # Alle Test-Frames des Folds — wie in der offiziellen gepoolten
        # Wertung zählen auch Auto-FPs aus den Rad-/Personen-Experimenten.
        # Dynamische Auto-GT existiert nur im Auto-Experiment (== Fold-Nr.).
        for result in predictions:
            sample_id = str(result.metainfo.get("sample_id", ""))
            sample = samples_by_id.get(sample_id)
            raw_entries = raw_lookup.get(sample_id)
            if sample is None or raw_entries is None:
                continue
            gts, flags, _ = static_flags_for_sample(sample, raw_entries)
            instance_pts = [instance.get("num_lidar_pts", -1)
                            for instance in sample["instances"]]
            gt_indices = [index for index, (label, _) in enumerate(gts)
                          if label == CAR]
            gt_boxes = np.asarray(
                [gts[index][1] for index in gt_indices]).reshape(-1, 7)
            gt_static = np.array([flags[index] for index in gt_indices])

            boxes = result.bboxes_3d.tensor.numpy()[:, :7]
            labels = np.asarray(result.labels_3d)
            scores = np.asarray(result.scores_3d)
            order = np.argsort(-scores[labels == CAR])
            car_boxes = boxes[labels == CAR][order]
            car_scores = scores[labels == CAR][order]
            ious = iou_matrix(car_boxes, gt_boxes, device)

            # (1)+(3): best prediction per GT instance
            for column, index in enumerate(gt_indices):
                gt_box = gts[index][1]
                best_iou = float(ious[:, column].max()) if len(car_boxes) \
                    else 0.0
                key = "static" if flags[index] else "dynamic"
                if flags[index]:
                    if best_iou < IOU_MAIN:
                        weak_static_pts.append(instance_pts[index])
                        weak_static_x.append(float(gt_box[0]))
                    else:
                        good_static_pts.append(instance_pts[index])
                else:
                    n_gt_dyn += 1
                if len(car_boxes):
                    k = int(np.argmax(ious[:, column]))
                    pred = car_boxes[k]
                    gt_l, gt_w = norm_dims(gt_box)
                    pr_l, pr_w = norm_dims(pred)
                    match_records[key].append(dict(
                        iou=best_iou, gt_len=gt_l, gt_wid=gt_w,
                        pred_len=pr_l, pred_wid=pr_w,
                        center_off=float(
                            np.linalg.norm(pred[:2] - gt_box[:2])),
                        score=float(car_scores[k])))
                else:
                    match_records[key].append(dict(
                        iou=0.0, gt_len=np.nan, gt_wid=np.nan,
                        pred_len=np.nan, pred_wid=np.nan,
                        center_off=np.nan, score=np.nan))

            # (2): replicate the dynamic AP60 matching, categorize FPs
            matched = [False] * len(gt_indices)
            for k in range(len(car_boxes)):
                if len(gt_indices):
                    m = int(np.argmax(ious[k]))
                    match_iou = float(ious[k, m])
                else:
                    m, match_iou = -1, 0.0
                score = float(car_scores[k])
                if match_iou >= IOU_MAIN and not matched[m]:
                    matched[m] = True
                    if not gt_static[m]:
                        tp_main += 1
                        for scores_out, tps_out in ranking.values():
                            scores_out.append(score)
                            tps_out.append(1)
                    continue  # true positive or ignored static match
                fp_scores.append(score)
                ranking["official"][0].append(score)
                ranking["official"][1].append(0)
                sloppy_static = (m >= 0 and gt_static[m]
                                 and match_iou >= IOU_SLOPPY
                                 and match_iou < IOU_MAIN)
                if not sloppy_static:
                    ranking["relaxed_ignore"][0].append(score)
                    ranking["relaxed_ignore"][1].append(0)
                if match_iou >= IOU_MAIN and matched[m]:
                    fp_categories["dup"] += 1
                elif sloppy_static:
                    fp_categories["sloppy_static"] += 1
                elif match_iou >= IOU_SLOPPY:
                    fp_categories["sloppy_dyn"] += 1
                else:
                    fp_categories["ghost"] += 1

    def summarize(records: list[dict]) -> dict:
        if not records:
            return {}
        arrays = {key: np.array([r[key] for r in records], dtype=float)
                  for key in records[0]}
        median = lambda x: float(np.nanmedian(x))  # noqa: E731
        return dict(
            n=len(records),
            iou_median=median(arrays["iou"]),
            share_iou60=float(np.mean(arrays["iou"] >= IOU_MAIN)),
            share_iou30=float(np.mean(arrays["iou"] >= 0.3)),
            gt_len_median=median(arrays["gt_len"]),
            gt_wid_median=median(arrays["gt_wid"]),
            d_len_median=median(arrays["pred_len"] - arrays["gt_len"]),
            d_wid_median=median(arrays["pred_wid"] - arrays["gt_wid"]),
            center_off_median=median(arrays["center_off"]),
            matched_score_median=median(arrays["score"]),
        )

    fp_scores_arr = np.asarray(fp_scores)
    return dict(
        dynamic=summarize(match_records["dynamic"]),
        static=summarize(match_records["static"]),
        ap60_ranking=dict(
            n_gt_dynamic=n_gt_dyn,
            true_positives=tp_main,
            false_positives=int(len(fp_scores_arr)),
            fp_score_ge_08=int((fp_scores_arr >= 0.8).sum()),
            fp_categories=fp_categories,
            ap60_official_replicated=ap11(
                ranking["official"][0], ranking["official"][1], n_gt_dyn),
            ap60_static_ignore_relaxed=ap11(
                ranking["relaxed_ignore"][0], ranking["relaxed_ignore"][1],
                n_gt_dyn),
        ),
        weak_static=dict(
            n=len(weak_static_pts),
            points_median=float(np.median(weak_static_pts))
            if weak_static_pts else None,
            good_points_median=float(np.median(good_static_pts))
            if good_static_pts else None,
            x_median=float(np.median(weak_static_x))
            if weak_static_x else None,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/exp"))
    parser.add_argument("--results-root", type=Path,
                        default=Path("results/crossval"))
    parser.add_argument("--raw-root", type=Path,
                        default=Path.home() / "data" / "experiments")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = {view: analyze_view(view, args.results_root, args.data_root,
                                 args.raw_root, args.device)
              for view in VIEWS}
    text = json.dumps(report, indent=1)
    print(text)
    if args.output is not None:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
