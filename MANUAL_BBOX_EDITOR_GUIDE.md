# Manual Bounding Box Editor Guide

This editor writes labels directly in the project JSON format:

```json
[
  {
    "label": "car",
    "bbox": [x, y, z, dx, dy, dz, yaw],
    "num_lidar_pts": 1234
  }
]
```

## Label Folder

Use only this final manual label folder for manually corrected merged labels:

```text
merged_labels_manual
```

Do not continue editing the temporary `merged_labels_clean_*` test folders.
They are useful only as historical experiments/reference.

## Recommended Start Command

From the project root:

```bash
python3 ./experiment/manual_bbox_editor.py \
  --experiment 1_experiment_car_1 \
  --pcd-dir merged_pcd \
  --label-dir merged_labels_manual \
  --class-name car \
  --start 180 \
  --template-from-index 272 \
  --topdown \
  --display-z-min 0.0
```

These settings:

- write to `merged_labels_manual`
- use the corrected car dimensions and center height (`z`) from absolute frame
  index `272`
- apply those dimensions and that center height to every loaded box, including
  boxes with previously saved manual dimensions
- show a provisional template-sized box for frames that do not have a label yet
- start in top-down view
- hide ceiling/high clutter for this dataset with `--display-z-min 0.0`

The reference frame was observed as:

```text
[93/192] 1760002170437337397 visible points_in_box=7683 center=(2.98,-0.15,0.68) dim=(4.90,1.80,1.30) yaw=-1.63
```

Because the editor had been started with `--start 180`, this relative frame
maps to absolute frame index:

```text
180 + (93 - 1) = 272
```

## Continue From Reference Frame

To continue directly from the reference frame:

```bash
python3 ./experiment/manual_bbox_editor.py \
  --experiment 1_experiment_car_1 \
  --pcd-dir merged_pcd \
  --label-dir merged_labels_manual \
  --class-name car \
  --start 272 \
  --template-from-index 272 \
  --topdown \
  --display-z-min 0.0
```

## Inspect Manual Labels

```bash
python3 ./experiment/pcd_sequence_viewer.py \
  --experiment 1_experiment_car_1 \
  --mode merged \
  --force-color \
  --show-labels \
  --label-dir merged_labels_manual \
  --point-size 3 \
  --start 180
```

## Controls

```text
Mouse       rotate / zoom / pan view
1/2/3       fine / normal / coarse edit steps
W/S         move bbox +Y / -Y
A/D         move bbox -X / +X
Q/E         move bbox up / down
J/L         rotate yaw left / right
U/O         length + / -
I/K         width  + / -
G/H         height + / -
P           copy previous frame label into current frame
C           create a new visible box and switch to free 3D edit camera
T           reset dimensions and center height to frame-272 template
V           show / hide current box without deleting file
Z           delete current frame label JSON and hide box
X           save current bbox JSON
N/B         save current frame, then next / previous frame
R           reset camera
F           free 3D edit camera centered on current box
M           top-down camera
?           print help
```

## Workflow

1. Start at the first frame that needs correction.
2. Use top-down view for the car trajectory; press `M` if the camera drifts.
3. If there is no useful box in the current frame, press `C` to create one.
   This also switches from top-down into the free 3D edit camera.
4. If a nearby previous label is good, press `P`. This copies the previous
   label into the current frame. With `--template-from-index`, dimensions and
   center height stay locked to the template.
5. Adjust position with `W/A/S/D` and `Q/E`.
6. Adjust yaw with `J/L`.
7. Press `T` if dimensions or center height drift; this restores the
   frame-272 car template geometry.
8. Press `N` for the next frame. It saves the current visible box first.

If you change dimensions with `U/O`, `I/K`, or `G/H`, the saved JSON is marked
with `manual_dims: true`. When `--template-from-index` is active, reopening the
editor still reapplies the template dimensions and template center height so the
car box stays the same size and on the same z-plane across frames.

Use `M` for top-down labeling and `F` for an oblique/drehbare 3D edit view
centered on the current box.

Use `1` for fine adjustments, `2` for normal adjustments, and `3` for coarse
adjustments. The default step sizes are now smaller than before:

```text
normal move step = 0.05 m
normal z step    = 0.025 m
normal dim step  = 0.05 m
normal yaw step  = 0.025 rad
```

By default, `N` and `B` save the current visible box before switching frames.
If the box is hidden with `V` or deleted with `Z`, frame switching does not
write a label for that frame.

Use `--no-autosave-on-frame-change` only if you want `N`/`B` to switch frames
without saving.

When `--template-from-index` is set, frames without an existing JSON label show
a provisional box using the template dimensions and template center height. If
you do not want that, add:

```bash
--no-default-box-for-empty
```
