# Code for Evaluating Experiment Data

In the experimental configuration, three scenarios were each executed thrice, yielding a total of nine experiments. Initially, a vehicle is utilized; secondly, a bicycle is employed; and thirdly, the author traverses the measurement scene on foot.

| Exp. ID | Main Object | Exp. (Folder) Name |
| --- | --- | --- |
| 1 | car | 1_experiment_car_1 |
| 2 | car | 2_experiment_car_2 |
| 3 | car | 3_experiment_car_3 |
| 4 | bycicle | 4_experiment_bike_1 |
| 5 | bycicle | 5_experiment_bike_3 |
| 6 | bycicle | 6_experiment_bike_4 |
| 7 | bycicle | 7_experiment_person_1 |
| 8 | bycicle | 8_experiment_person_2 |
| 9 | bycicle | 9_experiment_person_3 |
| | | bg-frames.tar.xz |


Note: An error occurred during bike_2, and therefore, the process had to be repeated.

## Dependencies

- Python 3.12
- numpy==2.3.3
- open3d==0.19.0
- pypcd4==1.3.0
- tqdm==4.67.1

## Preprocessing

First you need to extract the all 9 archives to your working directory.

```
tar -xvf X_experiment_Y_Z.tar.xz
```

Afterwards the `rosbag2` files need to be unpacked. Therefore we need a working ros2 kilted environment. (see https://docs.ros.org/en/jazzy/Releases/Release-Kilted-Kaiju.html)
Once activated run the script `pcd_export.sh` with uses internally `pcd_export.py`:

```
bash all_ros2bag_to_pcd_export.sh --base .
```
Note: This assumes that the nine experiment folder are in your current working directory.

## Point Cloud Transforming and Merging

Now the point clouds need to be transformated to align them with the axises and merge them. Therefore use the script `all_merge_pcds.sh`. This creates inside each experiment directory a folder `merged_pcd`, `os0_pcd_transform`, `os1_pcd_transform`. Internally it uses `pcd_merge.py` for processing the pointclouds.

```
bash all_merge_pcds.sh --base .
```
Note: This assumes that the nine experiment folder are in your current working directory.

## Labeling

The labeling process is done by removing a background frame and finding in the remaining point cloud the largest point clusters and annotate them to one of the thress classes `car`, `bike`, `person` by bounding box size templates. The process can be started using the script `auto_label_all.sh`. It uses internally the script `pcd_annotate.py`. This will create inside each experiment directory the folder `merged_labels`, `os0_labels` and `os1_labels`.

First extract the background frames into the work directory.

```
tar -xvf bg-frames.tar.xz
```

To label all experiments and the merged pointclouds as well as the single sensor ones run:

```
bash all_auto_label.sh .
```
Note: This assumes that the nine experiment folder are in your current working directory.

## Preparing for Object Detection

To pepare the files for usage with mmdetection3d (see https://mmdetection3d.readthedocs.io/en/latest/get_started.html) we need to call the script `create-data.py`.

Note: Changes to branch dev-1.x made for this thesis need to be loaded.

For working with a target dataset:

```
python tools/create_data.py exp --root-path data/exp/raw --out-dir data/exp --workers 4 --target-dataset [lumpi|osdar|kitti]
```

It will create three annotation files:
- `data/exp/exp_[TARGET-DATASET]_infos_merged.pkl` containing only the merged pointclouds and labels
- `data/exp/exp_[TARGET-DATASET]_infos_os0.pkl` containing only the sensor os0 pointclouds and labels
- `data/exp/exp_[TARGET-DATASET]_infos_os1.pkl` containing only the sensor os1 pointclouds and labels

The transformed data for the target dataset will be stored in

- `data/exp/points_[TARGET-DATASET]` for the point cloud data
- `data/exp/labels_[TARGET-DATASET]` for the label data

## Object Detection

Finally we can run the inference process using the pre-trained weight.

LUMPI:

```
python tools/test.py configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_exp.py PATH-TO-LUMPI-RESULT
```

KITTI:

```
python tools/test.py configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class-exp.py PATH-TO-KITTI-RESULT

```

OSDaR23:

```
python tools/test.py configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_osdar23.py PATH-TO-OSDAR23-RESULT

```

Results (mAP values and prediction) are stored in the following path.

LUMPI:

```
results/exp/exp_metric_test_TIMESTAMP.json
results/exp/exp_results_test_TIMESTAMP.pkl
```

KITTI:

```
results/exp/exp_kitti_metric_test_TIMESTAMP.json
results/exp/exp_kitti_results_test_TIMESTAMP.pkl
```

OSDaR23:

```
results/exp/exp_osdar_metric_test_TIMESTAMP.json
results/exp/exp_osdar_results_test_TIMESTAMP.pkl
```


## Visualization

For visualizing the results use the modified built-in open3d script. Besides the typical open3d controls (https://www.open3d.org/docs/release/tutorial/visualization/visualization.html) the following keys are registered:

| Key | Action |
| --- | --- |
| Space | Toggle Pause/Play if waiting time is > 0 |
| Right Arrow | Loads next sample |
| j | jumps to next added bounding box (first predictions than ground truths) |

```
python tools/misc/visualize_results.py PATH-TO-CONFIG --score-thr 0.0 --result PATH-TO-RESULT --wait-time -1
```
