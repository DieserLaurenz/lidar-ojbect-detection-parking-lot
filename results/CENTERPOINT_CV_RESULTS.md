# Cross-Validation Results

Experiment-held-out, paired-timestamp 3-fold evaluation. Values use dataset-level AP11 and the best validation checkpoint of each fold.

## Aggregate performance

| View | Fold 1 mAP | Fold 2 mAP | Fold 3 mAP | Mean +/- sample std | Pooled OOF mAP |
|---|---:|---:|---:|---:|---:|
| merged | 0.791 | 0.736 | 0.726 | 0.751 +/- 0.035 | 0.741 |
| os0 | 0.662 | 0.692 | 0.623 | 0.659 +/- 0.034 | 0.634 |
| os1 | 0.762 | 0.697 | 0.710 | 0.723 +/- 0.034 | 0.700 |

## Dynamic objects, pooled out-of-fold

| View | person AP30 / AP60 | bicycle AP30 / AP60 | car AP30 / AP60 |
|---|---:|---:|---:|
| merged | 0.892 / 0.365 (n=1772) | 0.690 / 0.366 (n=718) | 0.810 / 0.700 (n=339) |
| os0 | 0.558 / 0.157 (n=1127) | 0.685 / 0.168 (n=627) | 0.769 / 0.466 (n=339) |
| os1 | 0.885 / 0.306 (n=1679) | 0.564 / 0.317 (n=581) | 0.766 / 0.661 (n=339) |

## Static objects, pooled out-of-fold

| View | person AP30 / AP60 | bicycle AP30 / AP60 | car AP30 / AP60 |
|---|---:|---:|---:|
| merged | 0.908 / 0.217 (n=1246) | n/a / n/a (n=0) | 0.909 / 0.908 (n=12314) |
| os0 | 0.816 / 0.227 (n=1243) | n/a / n/a (n=0) | 0.906 / 0.898 (n=10812) |
| os1 | 0.907 / 0.190 (n=1199) | n/a / n/a (n=0) | 0.908 / 0.908 (n=11557) |

## Dynamic target class by held-out experiment

| Experiment | Target | merged AP30 / AP60 | os0 AP30 / AP60 | os1 AP30 / AP60 |
|---:|---|---:|---:|---:|
| 1 | car | 0.814 / 0.809 (n=105) | 0.910 / 0.847 (n=105) | 0.769 / 0.757 (n=105) |
| 2 | car | 0.818 / 0.675 (n=115) | 0.868 / 0.575 (n=115) | 0.678 / 0.517 (n=115) |
| 3 | car | 0.817 / 0.809 (n=119) | 0.845 / 0.667 (n=119) | 0.799 / 0.698 (n=119) |
| 4 | bicycle | 0.798 / 0.533 (n=130) | 0.696 / 0.240 (n=129) | 0.717 / 0.318 (n=99) |
| 5 | bicycle | 0.710 / 0.434 (n=310) | 0.709 / 0.173 (n=244) | 0.610 / 0.350 (n=306) |
| 6 | bicycle | 0.706 / 0.331 (n=278) | 0.681 / 0.143 (n=254) | 0.668 / 0.313 (n=176) |
| 7 | person | 0.896 / 0.353 (n=344) | 0.909 / 0.416 (n=340) | 0.838 / 0.286 (n=339) |
| 8 | person | 0.595 / 0.144 (n=315) | 0.792 / 0.239 (n=314) | 0.743 / 0.098 (n=306) |
| 9 | person | 0.906 / 0.355 (n=251) | 0.907 / 0.262 (n=251) | 0.813 / 0.206 (n=243) |

## Interpretation

- `merged` has the highest aggregate mAP in every fold; the mean advantage is about 0.092 over os0 and 0.028 over os1. With only three folds, this supports a balanced-performance claim, not a strong significance claim.
- Dynamic car AP30/AP60 is merged 0.810/0.700, os0 0.769/0.466, and os1 0.766/0.661. All views detect the moving car and merged is best at both thresholds. The old CenterPoint `dyn car ~0.03` collapse on the temporal split was a split artifact, matching the PointPillars os1 finding.
- Fusion's clearest benefits are person recall (merged 0.892 vs os0 0.558 dynamic AP30) and dynamic-car localization (AP60 0.700 vs 0.466/0.661).
- CenterPoint stays below PointPillars in every view (fold-mean mAP 0.751/0.659/0.723 vs 0.798/0.777/0.770), driven mainly by weaker AP60 localization. Under the label-based variant (section below) the AP60 deficit spans all three classes and is clearest for bicycles (-0.21 to -0.30 vs PointPillars).
- Dynamic car has exactly 339 paired GT instances in every view. Dynamic person/bicycle counts differ because annotations remain view-specific even though timestamps are paired; interpret those cross-view differences with that limitation.

## Label-based dynamic/static evaluation variant

Same named variant as for PointPillars
(`exp_eval_crossval.py --ignore-floor 0.25 --results-root results/crossval_cp_eval`):
predictions overlapping a non-target GT with IoU >= 0.25 are excluded from
the dynamic/static rankings using the manual static flags. Full report:
[`CENTERPOINT_CV_STATIC_AWARE.md`](CENTERPOINT_CV_STATIC_AWARE.md);
all-object metrics are bit-identical to the official ones. Key pooled
dynamic values, official -> label-based:

| View | dynamic mAP | person AP60 | car AP60 |
|---|---:|---:|---:|
| merged | 0.697 -> 0.716 | 0.365 -> 0.497 | 0.700 -> 0.707 |
| os0 | 0.486 -> 0.553 | 0.157 -> 0.315 | 0.466 -> 0.656 |
| os1 | 0.628 -> 0.650 | 0.306 -> 0.465 | 0.661 -> 0.669 |

Bicycle values are unchanged (no static bicycles). merged remains the best
CenterPoint view; CenterPoint stays below PointPillars in every class and
view at AP60 except person/os1 (0.465 vs 0.426).

## Consistency

The reproduced all-object fold mAP is compared with each metric JSON. The maximum absolute delta must remain below 1e-6.

Maximum reproduction delta: 3.331e-16

Pooled AP combines predictions from differently trained fold models; therefore per-fold values and their variation remain primary. The pooled value is an additional out-of-fold summary, not a replacement for fold reporting.

## Out-of-fold bicycle false positives

Predictions use score >= 0.3 and are false positives when no unmatched bicycle GT reaches IoU 0.3 in the same frame.

| View | FPs | FPs / 100 frames | Frames with FP | Isolated one-frame fraction | Median / p90 / max score |
|---|---:|---:|---:|---:|---:|
| merged | 239 | 11.26 | 223 | 57.7% | 0.471 / 0.791 / 0.903 |
| os0 | 77 | 3.63 | 77 | 42.9% | 0.677 / 0.875 / 0.935 |
| os1 | 137 | 6.46 | 136 | 34.3% | 0.670 / 0.862 / 0.927 |
