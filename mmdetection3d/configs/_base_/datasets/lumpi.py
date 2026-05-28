# dataset settings
dataset_type = 'LUMPIDataset'
data_root = 'data/lumpi'

classes = [
    "person",           # 0
    "car",              # 1
    "bicycle",          # 2
    "motorcycle",       # 3
    "bus",              # 4
    "truck",            # 5
    "unknown",          # 6
]

percentile = "95-5"

point_cloud_ranges = {
    # 95 and 5 percentile
    "95-5": [-47.33, -48.52, -2.38, 50.69, 46.35, 8.31],
}

point_cloud_range = point_cloud_ranges[percentile]

anchor_head_sizes = [
    # person
    [1.28, 0.77, 1.82],
    # car
    [2.5, 1.49, 1.52],
    # bicycle
    [1.86, 0.77, 1.81],
    # motorcycle
    [1.86, 0.78, 1.75],
    # bus
    [2.5, 1.51, 2.27],
    # truck
    [2.5, 1.67, 2.31],
    # unknown
    [2.25, 0.9, 1.58],
]

ann_files = {
    "train": "lumpi_infos_train.pkl",
    "val": "lumpi_infos_val.pkl",
    "test": "lumpi_infos_test.pkl",
}

point_cloud_dimension = 4
number_of_classes = len(classes)
input_modality = dict(use_lidar=True, use_camera=False)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=point_cloud_dimension,
        use_dim=point_cloud_dimension),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[0.25, 0.25, 0.1],
        global_rot_range=[-0.78539816, 0.78539816],
        rot_range=[-0.78539816, 0.78539816]),
    dict(
        type='RandomFlip3D',
        flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'box_mode_3d', 'box_type_3d', 'sample_id',
            'num_pts_feats', 'sample_idx', 'lidar_path',
        ],
    ),
]

val_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=point_cloud_dimension,
         use_dim=point_cloud_dimension
         ),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'box_mode_3d', 'box_type_3d', 'sample_id',
            'num_pts_feats', 'sample_idx', 'lidar_path',
        ],
    ),
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=ann_files['train'],
            data_prefix=dict(pts='points'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR',
            metainfo=dict(classes=classes),
        )
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='points'),
        ann_file=ann_files['val'],
        pipeline=val_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='lidar',
        metainfo=dict(classes=classes),
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='points'),
        ann_file=ann_files['test'],
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='lidar',
        metainfo=dict(classes=classes),
    ),
)

val_evaluator = dict(
    type='OSDaR23Metric',
    pklfile_prefix='results/lumpi/lumpi_results_train',
    result_prefix='results/lumpi/lumpi_metric_train',
)

test_evaluator = dict(
    type='OSDaR23Metric',
    pklfile_prefix='results/lumpi/lumpi_results_test',
    result_prefix='results/lumpi/lumpi_metric_test',
)
