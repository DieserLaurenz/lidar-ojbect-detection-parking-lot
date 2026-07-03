_base_ = ['./pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class-exp-os1.py']

# Ablation: GT-database ObjectSample (copy-paste augmentation) built from
# the os1 *train* split only, to counter the class imbalance
# (car:bicycle ~ 22:1). Pastes rare-class instances at their original
# scene positions from other train frames; collisions are rejected.

data_root = 'data/exp'
class_names = ['person', 'bicycle', 'car']
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + '/exp_kitti_dbinfos_os1_train.pkl',
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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

val_evaluator = dict(
    pklfile_prefix='results/exp/os1_gtsample_val_results',
    result_prefix='results/exp/os1_gtsample_val_metric',
)
test_evaluator = dict(
    pklfile_prefix='results/exp/os1_gtsample_test_results',
    result_prefix='results/exp/os1_gtsample_test_metric',
)
