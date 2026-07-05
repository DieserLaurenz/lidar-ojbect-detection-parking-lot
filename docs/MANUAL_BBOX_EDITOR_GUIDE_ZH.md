# 手动 3D Bounding Box 编辑器指南

这个编辑器会直接写入本项目使用的 JSON 标注格式：

```json
[
  {
    "label": "car",
    "bbox": [x, y, z, dx, dy, dz, yaw],
    "num_lidar_pts": 1234
  }
]
```

## 标注文件夹

手动修正后的 merged 点云标注只使用这个最终文件夹：

```text
merged_labels_manual
```

不要继续编辑临时的 `merged_labels_clean_*` 测试文件夹。那些文件夹只作为历史实验或参考。

## 推荐启动命令

在项目根目录运行：

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

这些参数的含义：

- 写入 `merged_labels_manual`
- 使用绝对帧索引 `272` 中已经修正好的车辆尺寸和中心高度（z）
- 对每个加载的 box 使用这个尺寸和中心高度，包括之前手动保存过尺寸的 box
- 启动时使用俯视图
- 用 `--display-z-min 0.0` 隐藏这个数据集中的天花板/高处干扰点

参考帧当时在终端中显示为：

```text
[93/192] 1760002170437337397 visible points_in_box=7683 center=(2.98,-0.15,0.68) dim=(4.90,1.80,1.30) yaw=-1.63
```

因为编辑器当时是用 `--start 180` 启动的，所以这个相对帧对应的绝对帧索引是：

```text
180 + (93 - 1) = 272
```

## 从参考帧继续编辑

如果要直接从参考帧继续：

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

## 查看手动标注结果

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

## 按键说明

```text
鼠标        旋转 / 缩放 / 平移视角
W/S         移动 bbox：+Y / -Y
A/D         移动 bbox：-X / +X
Q/E         移动 bbox：向上 / 向下
J/L         左右旋转 yaw
U/O         长度 + / -
I/K         宽度 + / -
G/H         高度 + / -
P           将上一帧的 label 复制到当前帧
C           新建一个可见 box，并切换到自由 3D 编辑视角
T           将尺寸和中心高度重置为第 272 帧的模板
V           显示 / 隐藏当前 box，不删除文件
Z           删除当前帧的 JSON label 文件并隐藏 box
X           保存当前 bbox JSON
N/B         保存当前帧，然后切换到下一帧 / 上一帧
R           重置相机
F           将自由 3D 编辑视角对准当前 box
M           切回俯视图
?           在终端打印帮助信息
```

## 推荐工作流程

1. 从需要修正的第一帧开始。
2. 车辆轨迹主要用俯视图编辑；如果视角跑偏，按 `M` 回到俯视图。
3. 如果当前帧没有可用的 box，按 `C` 新建一个 box。它会自动从俯视图切到可旋转的 3D 编辑视角。
4. 如果上一帧的 box 很好，按 `P`。这会把上一帧的 label 复制到当前帧；启用 `--template-from-index` 时，尺寸和中心高度仍然固定为模板值。
5. 用 `W/A/S/D` 和 `Q/E` 调整位置。
6. 用 `J/L` 调整 yaw。
7. 如果尺寸或中心高度变乱，按 `T` 恢复到第 272 帧的车辆模板。
8. 按 `N` 进入下一帧。切换前会先保存当前可见 box。

如果用 `U/O`、`I/K` 或 `G/H` 修改了尺寸，保存的 JSON 会带上：

```json
"manual_dims": true
```

启用 `--template-from-index` 时，之后重新打开编辑器仍会重新应用模板尺寸和模板中心高度，保证车辆 box 在所有帧中保持同样大小并处在同一 z 平面上。

用 `M` 做俯视图标注，用 `F` 切到以当前 box 为中心的斜视/可旋转 3D 编辑视角。

默认情况下，`N` 和 `B` 在切换帧之前会保存当前可见 box。如果当前 box 被 `V` 隐藏，或者用 `Z` 删除了 label 文件，那么切换帧时不会为这一帧写入 label。

如果你不想让 `N`/`B` 自动保存，可以启动时加：

```bash
--no-autosave-on-frame-change
```
