# Clean Auto-Labeling Guide

This guide creates new cleaned label folders without overwriting the original
auto-labels.

## What Changed

`experiment/pcd_annotate.py` now supports a cleaner labeling mode:

- `--no-static-labels`: disables hard-coded parked-car boxes.
- `--expected-class`: keeps only the expected dynamic class.
- `--use-template-dims`: stabilizes box sizes using class templates.
- `--keep-largest 1`: keeps only the largest plausible dynamic cluster.
- `--bg-threshold`: makes background removal less sensitive to small noise.
- `--min-cluster-points` and `--dbscan-eps`: tune DBSCAN per class.

The wrapper `experiment/all_auto_label_clean.sh` applies this to all
experiments and writes:

- `merged_labels_clean`
- `os0_labels_clean`
- `os1_labels_clean`

## Single Experiment Sanity Check

From the experiment data directory:

```bash
cd "$PROJ_DIR/data/2025_10_09/Experiment-Data/experiments"
```

Run clean labels for a small frame range first:

```bash
python3 "$PROJ_DIR/experiment/pcd_annotate.py" \
  1_experiment_car_1/merged_pcd \
  --label-dir merged_labels_clean_test \
  --bg-frame bg-frame-merged.pcd \
  --no-static-labels \
  --expected-class car \
  --use-template-dims \
  --keep-largest 1 \
  --bg-threshold 0.10 \
  --max-foreground-ratio 15 \
  --dbscan-eps 0.99 \
  --min-cluster-points 160 \
  --range 0:80 \
  --workers 1
```

Inspect the result from the project root:

```bash
cd "$PROJ_DIR"
python experiment/pcd_sequence_viewer.py \
  --experiment 1_experiment_car_1 \
  --mode merged \
  --force-color \
  --show-labels \
  --label-dir merged_labels_clean_test \
  --point-size 3
```

Compare with the original labels:

```bash
python experiment/pcd_sequence_viewer.py \
  --experiment 1_experiment_car_1 \
  --mode merged \
  --force-color \
  --show-labels \
  --label-dir merged_labels \
  --point-size 3
```

## All Experiments

From the experiment data directory:

```bash
cd "$PROJ_DIR/data/2025_10_09/Experiment-Data/experiments"
bash "$PROJ_DIR/experiment/all_auto_label_clean.sh" --base .
```

To run only one experiment:

```bash
bash "$PROJ_DIR/experiment/all_auto_label_clean.sh" \
  --base . \
  --include 1_experiment_car_1
```

To reduce CPU usage:

```bash
PARALLEL_JOBS=1 WORKERS_PER_JOB=2 \
bash "$PROJ_DIR/experiment/all_auto_label_clean.sh" --base .
```

## Inspect Clean Labels

Merged labels:

```bash
cd "$PROJ_DIR"
python experiment/pcd_sequence_viewer.py \
  --experiment 1_experiment_car_1 \
  --mode merged \
  --force-color \
  --show-labels \
  --label-dir merged_labels_clean \
  --point-size 3
```

Foreground plus clean labels:

```bash
python experiment/pcd_sequence_viewer.py \
  --experiment 1_experiment_car_1 \
  --mode foreground \
  --show-labels \
  --label-dir merged_labels_clean \
  --point-size 3
```

## Notes

If you rerun clean labeling after changing thresholds, remove or rename the old
`*_labels_clean` folders first. Otherwise old label files for frames that no
longer produce labels may remain in the folder.

The clean mode is intentionally conservative. It is better to keep fewer,
plausible labels than many noisy labels for fine-tuning.
