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
| merged | 0.901 / 0.561 (n=1772) | 0.800 / 0.668 (n=718) | 0.877 / 0.739 (n=339) |
| os0 | 0.653 / 0.284 (n=1127) | 0.895 / 0.378 (n=627) | 0.885 / 0.847 (n=339) |
| os1 | 0.896 / 0.276 (n=1679) | 0.744 / 0.529 (n=581) | 0.831 / 0.783 (n=339) |

## Static objects, pooled out-of-fold

| View | person AP30 / AP60 | bicycle AP30 / AP60 | car AP30 / AP60 |
|---|---:|---:|---:|
| merged | 0.901 / 0.229 (n=1246) | n/a / n/a (n=0) | 0.909 / 0.814 (n=12314) |
| os0 | 0.739 / 0.425 (n=1243) | n/a / n/a (n=0) | 0.908 / 0.908 (n=10812) |
| os1 | 0.889 / 0.111 (n=1199) | n/a / n/a (n=0) | 0.909 / 0.909 (n=11557) |

## Dynamic target class by held-out experiment

| Experiment | Target | merged AP30 / AP60 | os0 AP30 / AP60 | os1 AP30 / AP60 |
|---:|---|---:|---:|---:|
| 1 | car | 0.908 / 0.862 (n=105) | 0.921 / 0.921 (n=105) | 0.873 / 0.873 (n=105) |
| 2 | car | 0.895 / 0.761 (n=115) | 0.964 / 0.898 (n=115) | 0.768 / 0.768 (n=115) |
| 3 | car | 0.907 / 0.814 (n=119) | 0.942 / 0.900 (n=119) | 0.878 / 0.870 (n=119) |
| 4 | bicycle | 0.904 / 0.883 (n=130) | 0.906 / 0.561 (n=129) | 0.815 / 0.509 (n=99) |
| 5 | bicycle | 0.727 / 0.611 (n=310) | 0.907 / 0.425 (n=244) | 0.718 / 0.575 (n=306) |
| 6 | bicycle | 0.804 / 0.573 (n=278) | 0.893 / 0.266 (n=254) | 0.731 / 0.510 (n=176) |
| 7 | person | 0.908 / 0.440 (n=344) | 0.908 / 0.421 (n=340) | 0.899 / 0.372 (n=339) |
| 8 | person | 0.882 / 0.450 (n=315) | 0.885 / 0.417 (n=314) | 0.902 / 0.167 (n=306) |
| 9 | person | 0.902 / 0.399 (n=251) | 0.903 / 0.582 (n=251) | 0.905 / 0.259 (n=243) |

## Interpretation

- `merged` has the highest aggregate mAP in every fold, but the mean advantage is modest (about 0.021 over os0 and 0.028 over os1). With only three folds, this supports a balanced-performance claim, not a strong significance claim.
- Dynamic car AP30/AP60 is merged 0.877/0.739, os0 0.885/0.847, and os1 0.831/0.783. All views detect the moving car; os0 is slightly best. The old `os1=0.09 -> merged=0.90` headline was a temporal-split artifact. The merged AP60 deficit is a precision artifact caused by one sensor-adjacent parked car, not weaker localization of the moving car — see the decomposition section below.
- Fusion's clearest benefits are person localization and bicycle localization. os0 has higher bicycle AP30 but substantially lower AP60, while merged is more precise spatially.
- Dynamic car has exactly 339 paired GT instances in every view. Dynamic person/bicycle counts differ because annotations remain view-specific even though timestamps are paired; interpret those cross-view differences with that limitation.

## Dynamic-car AP60 decomposition (why merged trails)

Out-of-fold analysis pooled over all test frames — the exact population of
the official pooled AP60 (`tools/analysis_tools/exp_car_ap60_decomposition.py`,
added 2026-07-12). The replicated ranking reproduces the pooled dynamic-car
AP60 of every view exactly (delta 0.0), so these quantities decompose the
same computation that produced the table values above.

| Quantity | merged | os0 | os1 |
|---|---:|---:|---:|
| Median best IoU on the moving car | 0.848 | 0.840 | 0.837 |
| Moving-car frames with best IoU >= 0.6 | 91.4% | 97.3% | 88.5% |
| Median length error of the matched box | -0.01 m | -0.06 m | -0.09 m |
| Median BEV center offset | 0.13 m | 0.14 m | 0.14 m |
| FPs in the dynamic AP60 ranking with score >= 0.8 | 1,026 | 35 | 25 |
| ... of which IoU-0.25-0.6 boxes on parked cars (all scores) | 2,020 | 35 | 69 |
| Static car instances with best IoU < 0.6 | 2,035 / 12,314 | 45 / 10,812 | 94 / 11,557 |

How parked cars enter a dynamic-only metric: only the 339 moving-car GT
instances count as targets (recall denominator), but every car prediction
enters the ranking. A prediction is ignored only if it matches a static GT
with IoU >= threshold. At AP30 virtually all parked-car predictions clear
that bar and are ignored — dynamic-car AP30 is therefore unaffected. At
AP60, predictions on one sensor-adjacent parked car (x ~ 28.9 in the KITTI
frame — the same physical car in all three views; median ~18k lidar points
in merged; labeled 5.3 x 1.8 x 1.8 m) land at IoU 0.25-0.6 in nearly every
merged frame, stop being ignorable, and count as high-confidence false
positives of the dynamic evaluation, depressing precision at every recall
level.

Because the manual labels explicitly distinguish static from dynamic
objects, this attribution is available to the evaluation. A named variant
uses it: `exp_eval_crossval.py --ignore-floor 0.25` excludes predictions
overlapping a non-target GT with IoU >= 0.25 from the dynamic/static
rankings. Full report: [`CROSS_VALIDATION_STATIC_AWARE.md`](CROSS_VALIDATION_STATIC_AWARE.md)
(all-object metrics are bit-identical to the official ones; only the
dynamic/static sub-evaluations change). Key pooled dynamic values,
official -> label-based:

| View | dynamic mAP | person AP60 | car AP60 |
|---|---:|---:|---:|
| merged | 0.777 -> 0.836 | 0.561 -> 0.729 | 0.739 -> 0.883 |
| os0 | 0.700 -> 0.710 | 0.284 -> 0.370 | 0.847 -> 0.856 |
| os1 | 0.734 -> 0.756 | 0.276 -> 0.426 | 0.783 -> 0.789 |

Under the label-based rule the fusion leads AP60 in every class. The
presentation's dynamic-object slides use the label-based variant (with a
footnote); the official tables above remain the comparability reference.

merged boxes the moving car most accurately of all three views. The same
parked-car failure mode explains the merged static-car AP60 (0.814 vs
0.908/0.909). The dynamic-car AP60 column must therefore not be read as
fusion localizing the moving car worse. Open on-site question: what
actually stands at x ~ 28.9 next to the sensor (the 5.3 m long, 1.8 m tall
GT box suggests a van), and is its label correct?

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
