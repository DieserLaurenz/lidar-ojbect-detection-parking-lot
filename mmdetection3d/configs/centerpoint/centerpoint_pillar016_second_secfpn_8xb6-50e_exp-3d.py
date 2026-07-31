_base_ = [
    '../_base_/models/centerpoint_pillar02_second_secfpn_nus.py',
    '../_base_/schedules/cyclic-20e.py',
    '../_base_/default_runtime.py',
]

# CenterPoint-Pillar auf den Experiment-Daten (Tiefgarage), Daten v2.
# Vergleichslauf zu pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class-exp.py:
# gleicher Pillar-Encoder-Ansatz und dasselbe 0.16-m-Raster (432x496),
# aber anchor-freier CenterHead statt Anchor-Head. Start vom
# nuScenes-Pretraining (es gibt keinen offiziellen KITTI-CenterPoint).

dataset_type = 'LUMPIDataset'
data_root = 'data/exp'
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
voxel_size = [0.16, 0.16, 4]
class_names = ['person', 'bicycle', 'car']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None
load_from = ('checkpoints/centerpoint_02pillar_second_secfpn_circlenms'
             '_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth')

model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(
            max_num_points=32,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000),
            point_cloud_range=point_cloud_range)),
    pts_voxel_encoder=dict(
        in_channels=4,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(output_shape=(496, 432)),
    pts_bbox_head=dict(
        tasks=[dict(num_class=3, class_names=class_names)],
        # kein vel-Head: unsere GT-Boxen sind 7-dim (ohne Geschwindigkeit);
        # _delete_ noetig, sonst bleibt vel aus der Basis-Config erhalten
        common_heads=dict(
            _delete_=True, reg=(2, 2), height=(1, 2), dim=(3, 2),
            rot=(2, 2)),
        bbox_coder=dict(
            pc_range=point_cloud_range[:2],
            post_center_range=[-10, -49.68, -10, 79.12, 49.68, 10],
            voxel_size=voxel_size[:2],
            code_size=7)),
    train_cfg=dict(
        pts=dict(
            grid_size=[432, 496, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-10, -49.68, -10, 79.12, 49.68, 10],
            min_radius=[4],
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2])))

ann_files = {
    "merged": {
        "train": "exp_kitti_infos_merged_train.pkl",
        "val": "exp_kitti_infos_merged_val.pkl",
        "test": "exp_kitti_infos_merged_test.pkl",
    },
    "os0": {
        "train": "exp_kitti_infos_os0_train.pkl",
        "val": "exp_kitti_infos_os0_val.pkl",
        "test": "exp_kitti_infos_os0_test.pkl",
    },
    "os1": {
        "train": "exp_kitti_infos_os1_train.pkl",
        "val": "exp_kitti_infos_os1_val.pkl",
        "test": "exp_kitti_infos_os1_test.pkl",
    },
}
view = "merged"
train_ann_file = ann_files[view]["train"]
val_ann_file = ann_files[view]["val"]
test_ann_file = ann_files[view]["test"]

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

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
             'box_mode_3d', 'box_type_3d', 'sample_id',
             'num_pts_feats', 'sample_idx', 'lidar_path',
        ],
    ),
]

custom_hooks = [
    dict(type='InferenceTimeHook'),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='osdar23/mAP',
        rule='greater',
    ),
)

train_dataloader = dict(

    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        data_prefix=dict(pts='points_kitti'),
        modality=input_modality,
        test_mode=False,
        box_type_3d='lidar',
        backend_args=backend_args,
    ),
)

val_dataloader = dict(

    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        data_prefix=dict(pts='points_kitti'),
        modality=input_modality,
        test_mode=True,
        box_type_3d='lidar',
        backend_args=backend_args,
    ),
)

test_dataloader = dict(

    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=test_ann_file,
        pipeline=test_pipeline,
        data_prefix=dict(pts='points_kitti'),
        modality=input_modality,
        test_mode=True,
        box_type_3d='lidar',
        backend_args=backend_args,
    ),
)

val_evaluator = dict(

    type='OSDaR23Metric',
    pklfile_prefix='results/exp_cp/exp_kitti_results_val',
    result_prefix='results/exp_cp/exp_kitti_metric_val',
    collect_device='gpu',
    dump_only=False,
    use_kitti=True,
    score_threshold=0.0,
    max_workers=1,
)
test_evaluator = dict(

    type='OSDaR23Metric',
    pklfile_prefix='results/exp_cp/exp_kitti_results_test',
    result_prefix='results/exp_cp/exp_kitti_metric_test',
    collect_device='gpu',
    dump_only=False,
    use_kitti=True,
    score_threshold=0.0,
    max_workers=1,
)

lr = 0.0001
optim_wrapper = dict(optimizer=dict(lr=lr))
epoch_num = 50
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = dict()
