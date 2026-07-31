# Cross-Validation Results

Experiment-held-out, paired-timestamp 3-fold evaluation. Values use dataset-level AP11 and the best validation checkpoint of each fold.

## Aggregate performance

| View | Fold 1 mAP | Fold 2 mAP | Fold 3 mAP | Mean +/- sample std | Pooled OOF mAP |
|---|---:|---:|---:|---:|---:|
| merged | 0.840 | 0.786 | 0.768 | 0.798 +/- 0.037 | 0.789 |
| os0 | 0.793 | 0.781 | 0.757 | 0.777 +/- 0.018 | 0.759 |
| os1 | 0.801 | 0.751 | 0.759 | 0.770 +/- 0.027 | 0.765 |

## Dynamic objects, pooled out-of-fold

| View | person AP30 / AP60 | bicycle AP30 / AP60 | car AP30 / AP60 |
|---|---:|---:|---:|
| merged | 0.901 / 0.729 (n=1772) | 0.800 / 0.668 (n=718) | 0.891 / 0.883 (n=339) |
| os0 | 0.653 / 0.370 (n=1127) | 0.895 / 0.378 (n=627) | 0.885 / 0.856 (n=339) |
| os1 | 0.896 / 0.426 (n=1679) | 0.744 / 0.529 (n=581) | 0.832 / 0.789 (n=339) |

## Static objects, pooled out-of-fold

| View | person AP30 / AP60 | bicycle AP30 / AP60 | car AP30 / AP60 |
|---|---:|---:|---:|
| merged | 0.902 / 0.283 (n=1246) | n/a / n/a (n=0) | 0.909 / 0.814 (n=12314) |
| os0 | 0.741 / 0.460 (n=1243) | n/a / n/a (n=0) | 0.908 / 0.908 (n=10812) |
| os1 | 0.892 / 0.163 (n=1199) | n/a / n/a (n=0) | 0.909 / 0.909 (n=11557) |

## Dynamic target class by held-out experiment

| Experiment | Target | merged AP30 / AP60 | os0 AP30 / AP60 | os1 AP30 / AP60 |
|---:|---|---:|---:|---:|
| 1 | car | 0.908 / 0.908 (n=105) | 0.921 / 0.921 (n=105) | 0.873 / 0.873 (n=105) |
| 2 | car | 0.898 / 0.813 (n=115) | 0.964 / 0.898 (n=115) | 0.768 / 0.768 (n=115) |
| 3 | car | 0.907 / 0.907 (n=119) | 0.942 / 0.900 (n=119) | 0.878 / 0.870 (n=119) |
| 4 | bicycle | 0.904 / 0.883 (n=130) | 0.906 / 0.561 (n=129) | 0.815 / 0.509 (n=99) |
| 5 | bicycle | 0.727 / 0.611 (n=310) | 0.907 / 0.425 (n=244) | 0.718 / 0.575 (n=306) |
| 6 | bicycle | 0.804 / 0.573 (n=278) | 0.893 / 0.266 (n=254) | 0.731 / 0.510 (n=176) |
| 7 | person | 0.908 / 0.473 (n=344) | 0.908 / 0.462 (n=340) | 0.899 / 0.394 (n=339) |
| 8 | person | 0.882 / 0.611 (n=315) | 0.885 / 0.631 (n=314) | 0.902 / 0.387 (n=306) |
| 9 | person | 0.902 / 0.745 (n=251) | 0.903 / 0.716 (n=251) | 0.905 / 0.576 (n=243) |

## Interpretation

- `merged` has the highest aggregate mAP in every fold, but the mean advantage is modest (about 0.021 over os0 and 0.028 over os1). With only three folds, this supports a balanced-performance claim, not a strong significance claim.
- Dynamic car AP30/AP60 is merged 0.891/0.883, os0 0.885/0.856, and os1 0.832/0.789. All views detect the moving car; os0 is slightly best. The old `os1=0.09 -> merged=0.90` headline was a temporal-split artifact.
- Fusion's clearest benefits are person localization and bicycle localization. os0 has higher bicycle AP30 but substantially lower AP60, while merged is more precise spatially.
- Dynamic car has exactly 339 paired GT instances in every view. Dynamic person/bicycle counts differ because annotations remain view-specific even though timestamps are paired; interpret those cross-view differences with that limitation.

## Consistency

The reproduced all-object fold mAP is compared with each metric JSON. The maximum absolute delta must remain below 1e-6.

Maximum reproduction delta: 2.220e-16

Pooled AP combines predictions from differently trained fold models; therefore per-fold values and their variation remain primary. The pooled value is an additional out-of-fold summary, not a replacement for fold reporting.

## Out-of-fold bicycle false positives

Predictions use score >= 0.3 and are false positives when no unmatched bicycle GT reaches IoU 0.3 in the same frame.

| View | FPs | FPs / 100 frames | Frames with FP | Isolated one-frame fraction | Median / p90 / max score |
|---|---:|---:|---:|---:|---:|
| merged | 315 | 14.84 | 283 | 50.8% | 0.394 / 0.627 / 0.955 |
| os0 | 157 | 7.40 | 156 | 47.1% | 0.419 / 0.676 / 0.906 |
| os1 | 372 | 17.53 | 356 | 41.1% | 0.375 / 0.746 / 0.971 |
