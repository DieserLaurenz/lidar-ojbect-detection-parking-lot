# Experiment-held-out Cross-Validation

**Protocol frozen:** 2026-07-11, before inspecting any cross-validation test
result.

## Purpose

The previous temporal 80/10/10 split confounded test membership with the last
object positions in every trajectory. This protocol evaluates generalization
to complete unseen recordings and uses identical physical timestamps for
`merged`, `os0`, and `os1`.

## Folds

| Fold | Held-out car | Held-out bicycle | Held-out person |
|---|---:|---:|---:|
| 1 | experiment 1 | experiment 4 | experiment 7 |
| 2 | experiment 2 | experiment 5 | experiment 8 |
| 3 | experiment 3 | experiment 6 | experiment 9 |

Only the intersection of valid timestamps across all three views is used:
2,122 paired frames. View-specific labels remain unchanged. The complete
held-out experiments form the test split. For each remaining experiment, the
last 10% form validation; ten preceding frames are excluded as a temporal
guard. Validation is used only for checkpoint selection.

| Fold | Train | Validation | Test | Guard/excluded |
|---|---:|---:|---:|---:|
| 1 | 1,239 | 146 | 677 | 60 |
| 2 | 1,172 | 137 | 753 | 60 |
| 3 | 1,227 | 143 | 692 | 60 |

## Training

- Architecture: PointPillars, initialized from the KITTI three-class
  checkpoint.
- Configuration: the existing 50-epoch experiment fine-tuning setup with
  GT-Sampling (`person=8`, `bicycle=10`).
- GT databases are rebuilt separately from each fold's train PKL. No held-out
  experiment occurs in a train PKL or sampling database.
- Fixed seed: 42.
- Nine runs: 3 folds x 3 views.
- Checkpoint selection: highest validation `osdar23/mAP` only.
- Final test invocation is automatic after every training run.

Server paths:

```text
~/runs/pointpillars_crossval/<view>_cv<fold>_gtsample/
~/runs/pointpillars_crossval/<view>_cv<fold>_gtsample.log
```

Reproduction:

```bash
cd ~/projects/dcaiti_masterarbeit/mmdetection3d
source ~/miniforge3/bin/activate mmdet3d
./tools/run_exp_crossval.sh prepare
./tools/run_exp_crossval.sh launch
```

## Planned reporting

1. Report every fold separately.
2. Report mean and standard deviation across folds.
3. Pool all out-of-fold predictions once and compute dataset-level AP.
4. Evaluate dynamic and static objects separately.
5. Compare views on the paired timestamps and report per-experiment results.

The protocol still uses one physical scene and mostly the same physical object
instances. It tests generalization to unseen recordings/trajectories in that
scene, not generalization to a new garage or object population.

## Completed results (2026-07-11)

All nine runs and their automatic best-checkpoint tests completed. The full
machine-readable and Markdown reports are:

```text
results/CROSS_VALIDATION_RESULTS.json
results/CROSS_VALIDATION_RESULTS.md
```

### Aggregate test performance

| View | Fold 1 | Fold 2 | Fold 3 | Fold mean +/- sample std | Pooled OOF |
|---|---:|---:|---:|---:|---:|
| merged | 0.8396 | 0.7855 | 0.7685 | **0.7979 +/- 0.0371** | **0.789** |
| os0 | 0.7930 | 0.7810 | 0.7574 | 0.7772 +/- 0.0181 | 0.759 |
| os1 | 0.8014 | 0.7508 | 0.7587 | 0.7703 +/- 0.0272 | 0.765 |

`merged` has the highest aggregate mAP in every fold, but the advantage is
modest. With only three folds, report the individual values rather than making
a strong significance claim.

### Dynamic objects, pooled OOF AP30 / AP60

| View | person | bicycle | car |
|---|---:|---:|---:|
| merged | **0.901 / 0.561** | 0.800 / **0.668** | 0.877 / 0.739 |
| os0 | 0.653 / 0.284 | **0.895** / 0.378 | **0.885 / 0.847** |
| os1 | 0.896 / 0.276 | 0.744 / 0.529 | 0.831 / 0.783 |

The old dynamic-car result `os1=0.09 -> merged=0.90` was a temporal-split
artifact and must no longer be presented as the main result. On complete
held-out recordings all views detect the moving car well. `os0` is slightly
best for dynamic car; fusion is not necessary for that target. Fusion's
defensible benefit is the best balanced aggregate performance, especially
person AP/localization and bicycle localization, not a universal car rescue.

### Out-of-fold bicycle false positives

At score >= 0.3 and IoU 0.3:

| View | FPs | FPs / 100 frames | Isolated one-frame fraction |
|---|---:|---:|---:|
| merged | 315 | 14.84 | 50.8% |
| os0 | 157 | 7.40 | 47.1% |
| os1 | 372 | 17.53 | 41.1% |

The largest clusters repeat at fixed coordinates across multiple held-out
experiments. This strengthens the scene-position/background-memorization
diagnosis. Most FP scores are low (median 0.375--0.419), but a few high-score
errors remain, so thresholding alone is insufficient.

The analysis reproduced all nine official fold mAP values with maximum
absolute error `2.22e-16`; no raw-label static flag was unmatched.
