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
| merged | 0.892 / 0.497 (n=1772) | 0.690 / 0.366 (n=718) | 0.810 / 0.707 (n=339) |
| os0 | 0.558 / 0.315 (n=1127) | 0.685 / 0.168 (n=627) | 0.816 / 0.656 (n=339) |
| os1 | 0.885 / 0.465 (n=1679) | 0.564 / 0.317 (n=581) | 0.767 / 0.669 (n=339) |

## Static objects, pooled out-of-fold

| View | person AP30 / AP60 | bicycle AP30 / AP60 | car AP30 / AP60 |
|---|---:|---:|---:|
| merged | 0.908 / 0.233 (n=1246) | n/a / n/a (n=0) | 0.909 / 0.908 (n=12314) |
| os0 | 0.816 / 0.246 (n=1243) | n/a / n/a (n=0) | 0.906 / 0.899 (n=10812) |
| os1 | 0.908 / 0.234 (n=1199) | n/a / n/a (n=0) | 0.908 / 0.908 (n=11557) |

## Dynamic target class by held-out experiment

| Experiment | Target | merged AP30 / AP60 | os0 AP30 / AP60 | os1 AP30 / AP60 |
|---:|---|---:|---:|---:|
| 1 | car | 0.814 / 0.809 (n=105) | 0.910 / 0.847 (n=105) | 0.769 / 0.757 (n=105) |
| 2 | car | 0.818 / 0.675 (n=115) | 0.868 / 0.575 (n=115) | 0.678 / 0.517 (n=115) |
| 3 | car | 0.817 / 0.809 (n=119) | 0.845 / 0.667 (n=119) | 0.799 / 0.698 (n=119) |
| 4 | bicycle | 0.798 / 0.533 (n=130) | 0.696 / 0.240 (n=129) | 0.717 / 0.318 (n=99) |
| 5 | bicycle | 0.710 / 0.434 (n=310) | 0.709 / 0.173 (n=244) | 0.610 / 0.350 (n=306) |
| 6 | bicycle | 0.706 / 0.331 (n=278) | 0.681 / 0.143 (n=254) | 0.668 / 0.313 (n=176) |
| 7 | person | 0.896 / 0.453 (n=344) | 0.909 / 0.480 (n=340) | 0.838 / 0.372 (n=339) |
| 8 | person | 0.595 / 0.394 (n=315) | 0.796 / 0.421 (n=314) | 0.743 / 0.293 (n=306) |
| 9 | person | 0.906 / 0.756 (n=251) | 0.907 / 0.584 (n=251) | 0.813 / 0.493 (n=243) |

## Interpretation

- `merged` has the highest aggregate mAP in every fold, but the mean advantage is modest (about 0.021 over os0 and 0.028 over os1). With only three folds, this supports a balanced-performance claim, not a strong significance claim.
- Dynamic car AP30/AP60 is merged 0.810/0.707, os0 0.816/0.656, and os1 0.767/0.669. All views detect the moving car; os0 is slightly best. The old `os1=0.09 -> merged=0.90` headline was a temporal-split artifact.
- Fusion's clearest benefits are person localization and bicycle localization. os0 has higher bicycle AP30 but substantially lower AP60, while merged is more precise spatially.
- Dynamic car has exactly 339 paired GT instances in every view. Dynamic person/bicycle counts differ because annotations remain view-specific even though timestamps are paired; interpret those cross-view differences with that limitation.

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
