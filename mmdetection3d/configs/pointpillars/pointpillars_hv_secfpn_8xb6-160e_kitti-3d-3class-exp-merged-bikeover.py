_base_ = ['./pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class-exp-merged.py']

# Ablation: repeat-factor frame oversampling (Gupta et al., LVIS) via
# ClassBalancedDataset. With oversample_thr=1.0 a frame is repeated by
# max_c sqrt(1 / f(c)) where f(c) is the fraction of frames containing
# class c; bicycle frames (f ~ 0.26) are repeated ~2x per epoch.

dataset_type = 'LUMPIDataset'
data_root = 'data/exp'
class_names = ['person', 'bicycle', 'car']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
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
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1.0,
        dataset=dict(
            type=dataset_type,
            metainfo=metainfo,
            data_root=data_root,
            ann_file='exp_kitti_infos_merged_train.pkl',
            pipeline=train_pipeline,
            data_prefix=dict(pts='points_kitti'),
            modality=input_modality,
            test_mode=False,
            box_type_3d='lidar',
            backend_args=backend_args,
        ),
    ),
)

val_evaluator = dict(
    pklfile_prefix='results/exp/merged_bikeover_val_results',
    result_prefix='results/exp/merged_bikeover_val_metric',
)
test_evaluator = dict(
    pklfile_prefix='results/exp/merged_bikeover_test_results',
    result_prefix='results/exp/merged_bikeover_test_metric',
)
