# MMDetection3D

This code is based on [open-mmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/962f093736ffe55c089bc618842a8b8567318c8c), branch `dev-1.x`, commit `962f093736ffe55c089bc618842a8b8567318c8c`.

## Full Diff

## Major Contribution
 - adding dataset converters (`tools/create-data.py`, `tools/dataset_converters/update_infos_to_v2.py`, `tools/dataset_converters/create_gt_database.py`)
    - OSDaR23
    - LUMPI
    - Experiment Data
- Datasets (`mmdetection3d/datasets`)
    - OSDaR23
    - LUMPI
- Neural Network Configuration (`configs`)
    - PointPillars
        - OSDaR23
        - LUMPI
- Hooks
    - StopOnNaNHook
    - InferenceTimeHook
- Evaluation/Metric
    - OSDaR23Metric (including tests)
- Tools
    - DatasetAnalyzer
        - OSDaR23
        - LUMPI
        - KITTI
        - Experiment


For full change see Path `changes_thea_pagel.patch`

Adujstments:
- Visualizer `/mmdet3d/visualization/local_visualizer.py`
- Training Script `/tools/dist_train.sh`
- Visualize Results `tools/misc/visualize_results.py`

## Installation

For setting up the mmdetection3d environment follow the instructions here: https://mmdetection3d.readthedocs.io/en/latest/get_started.html

Make sure to use:

- Pyhon 3.8
- conda 24.7.1
- CUDA 12.2
- mmcv>=2.0.0rc4
- mmdet>=3.0.0
- and instead of `git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x` this repository!

## Structure

The following folders in the mmdetection3d parent directory are important

- `configs` - containing all network configuration files for running any kind of interaction
- `data` - source directory for all datasets
- `mmdetection3d` - directory for custom hooks, datasets or transformation
- `tests` - test cases, e.g. for testing evaluation/metric scripts
- `tools` - binaries or python files for starting training or tests or debug method for visualization and logging
- `results` - storage path for evaluation results of OSDaR23Metric
- `work_dir` - directory of logs, checkpoints

## Dataset Converters

To pepare the datasets for being processed by the neural networks they need to be converted. The central script for this step is `tools/create-data.py`. The call looks like this:

```
# OSDaR23
python tools/create_data.py exp --root-path data/osdar23/raw --out-dir data/osdar23 --workers 4

# LUMPI
python tools/create_data.py exp --root-path data/lumpi/raw --out-dir data/lumpi --workers 4

# Experiment data
python tools/create_data.py exp --root-path data/exp/raw --out-dir data/exp --workers 4 --target-dataset {lumpi,kitti,osdar}

```

To make this work the raw data are extracted into `data/[DATASET]/raw`. The resulting files and folders are

- `points` - containing all pointcloud data in binary format
- `labels` - containing all annotation data in json format
- `[DATASET]_infos_train.pkl` - containing all training set annotation and pointcloud binary name
- `[DATASET]_infos_val.pkl` - containing all validation set annotation and pointcloud binary name
- `[DATASET]_infos_test.pkl` - containing all test set annotation and pointcloud binary name

The following contributions had been made:
- `tools/create-data.py`
    - add calls for OSDaR23 dataset
    - add calls for LUMPI dataset
    - add calls for Experiment data
- `tools/dataset_converters/osdar23.py`
    - add parser for pointcloud data and export to bin (format: x,y,z,intensity; float32)
    - add parser for labels (ASAM openLabel to json open-mmlab instance format)
    - add training, validation and test set generator
- `tools/dataset_converters/lumpi.py`
    - add parser for pointcloud data and export to bin (format: x,y,z,intensity; float32)
    - add parser for labels (csv to json open-mmlab instance format)
    - add training, validation and test set generator
- `tools/dataset_converters/exp.py`
    - add parser for pointcloud data and export to bin (format: x,y,z,intensity; float32)
    - add parser for labels (json to json open-mmlab instance format)
    - add set generator for merged and single sensor os0 and os1
- `tools/dataset_converters/update_infos_to_v2.py`
    - add calls for OSDaR23 dataset to pack annotation files (train, test, val)
    - add calls for LUMPI dataset to pack annotation files (train, test, val)
    - add calls for Experiment data to pack annotation files (train, test, val)
- `tools/dataset_converters/create_gt_database.py`
    - add calls for OSDaR23 to create groundtruth datasbase
    - add calls for LUMPI to create groundtruth datasbase
- `tools/dataset_converters/exp_to_kitti.py`
    - to make Experiment data work with KITTI network it needed to be transformed into correct world space
- `tools/dataset_converters/exp_to_lumpi.py`
    - to make Experiment data work with LUMPI network it needed to be transformed into correct world space
- `tools/dataset_converters/exp_to_osdar23.py`
    - to make Experiment data work with OSDaR23 network it needed to be transformed into correct world space

## Datasets

To load and parse the annotation data and meta inforamtion correctly mmdetection3d needs adjustments to their basedata set. Therefore the following files were created inhereting Det3DDataset.

- `mdet3d/datasets/osdar23.py` - Dataset to load OSDaR23 dataset
- `mdet3d/datasets/lumpi.py` - Dataset to load LUMPI dataset

Major changes are:

- creation of LiDARInstance3DBoxes for annotation data
- adding sample_id besides samples_idx to identify frame and experiment
- load axis_align_matrix
- getter function by sample_id

## Neural Network Configuration

To configure mmdetection3d provides the following structure:
- config files can include multiple other config files using `_base_ = [...]`
- dict entries can be overwritten partially or totally
- the folder `configs/_base_` contains basic settings for `datasets` and `model`
- they are combined with runtime settings in the main `config` dir

Major additions:
- `configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_osdar23.py`
- `configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_lumpi.py`
- `configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_exp.py`
- `configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class-exp.py`

- `configs/_base_/models/pointpillars_hv_secfpn_osdar23.py`
- `configs/_base_/models/pointpillars_hv_secfpn_lumpi.py`
- `configs/_base_/datasets/osdar23.py`
- `configs/_base_/datasets/lumpi.py`

## Hooks

The following hooks were added to make mmdetection3d work more suitable for our use case.

- `mmdet3d/engine/hooks/stop_on_nan_hook.py` - StopOnNaNHook stops the training process if the loss gets numerically instable and results in NaN (e.g. too small anchor sizes)
- `mmdet3d/engine/hooks/log_time_hook.py` - InferenceTimeHook measures and logs time it takes for inference progress for a single sample

## Evaluation/Metric

To test the performace a custom metric was build using IoU to calculate a mAP on different IoU thresholds. See thesis Chapter Implementation for more details.

- `mmdet3d/evaluation/metrics/osdar23_metric.py` - OSDaR23Metric calculating mAP with IoU
- `tests/test_evaluation/test_metrics/test_osdar23_metric.py` - tests the functionality of OSDaR23Metric
- `tests/data/osdar23` - containg all the data needed for the etst cases

Although the same Metric is used for LUMPI and the experiment data it is called OSDaR23Metric because it was implemented primarly for this dataset.

## Tools

To determine the right configuration for each dataset an analyzer was written to gather needed information.

- `tools/analysis_tools/dataset_analyzer.py` - Base Analyzer which needs to be inherited and adopted for each dataset
- `tools/analysis_tools/dataset_analyzer_osdar23.py` - changes on the base analyzer for OSDaR23 dataset
- `tools/analysis_tools/dataset_analyzer_lumpi.py` - changes on the base analyzer for LUMPI dataset
- `tools/analysis_tools/dataset_analyzer_kitti.py` - changes on the base analyzer for KITTI dataset
- `tools/analysis_tools/dataset_analyzer_exp.py` - changes on the base analyzer for Experiment dataset

Metric the analyzer is able to gather:
- pointcloud range
- anchor sizes (grouped by class or on whole dataset, different numbers per class)
- voxel_size and voxel_numbers / points_per_voxel
