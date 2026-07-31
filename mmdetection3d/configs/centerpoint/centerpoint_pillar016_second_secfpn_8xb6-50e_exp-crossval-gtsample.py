import os


_base_ = [
    './centerpoint_pillar016_second_secfpn_8xb6-50e_exp-3d.py',
]

# CenterPoint unter dem eingefrorenen Cross-Validation-Protokoll
# (results/CROSS_VALIDATION.md): identische Folds, Splits, GT-Sampling-
# Policy und Seed wie der PointPillars-Lauf, nur die Architektur
# (anchor-freier CenterHead) unterscheidet sich.

fold = int(os.environ.get('EXP_CV_FOLD', '1'))
view = os.environ.get('EXP_CV_VIEW', 'merged')
if fold not in (1, 2, 3):
    raise ValueError(f'EXP_CV_FOLD must be 1, 2, or 3; got {fold}')
if view not in ('merged', 'os0', 'os1'):
    raise ValueError(f'EXP_CV_VIEW must be merged, os0, or os1; got {view}')

data_root = 'data/exp'
class_names = ['person', 'bicycle', 'car']
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
backend_args = None
run_name = f'cp_{view}_cv{fold}_gtsample'

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + f'/exp_kitti_dbinfos_{view}_cv{fold}_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(person=10, bicycle=10)),
    classes=class_names,
    sample_groups=dict(person=8, bicycle=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]

train_dataloader = dict(
    dataset=dict(
        ann_file=f'exp_kitti_infos_{view}_cv{fold}_train.pkl',
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(ann_file=f'exp_kitti_infos_{view}_cv{fold}_val.pkl'))
test_dataloader = dict(
    dataset=dict(ann_file=f'exp_kitti_infos_{view}_cv{fold}_test.pkl'))

val_evaluator = dict(
    pklfile_prefix=f'results/crossval_cp/{run_name}_val_results',
    result_prefix=f'results/crossval_cp/{run_name}_val_metric')
test_evaluator = dict(
    pklfile_prefix=f'results/crossval_cp/{run_name}_test_results',
    result_prefix=f'results/crossval_cp/{run_name}_test_metric')

randomness = dict(seed=42, deterministic=False)
