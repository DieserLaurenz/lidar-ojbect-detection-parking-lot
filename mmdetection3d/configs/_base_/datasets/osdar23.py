# dataset settings
dataset_type = 'OSDaR23Dataset'
data_root = 'data/osdar23/'

classes = [
        "person",           # 0
        "crowd",            # 1
        "train",            # 2
        "wagons",           # 3
        "bicycle",          # 4
        "road_vehicle",     # 5
        "animal",           # 6
        "switch",           # 7
        "catenary_pole",    # 8
        "signal_pole",      # 9
        "signal",           # 10
        "signal_bridge",    # 11
        "buffer_stop",      # 12
    ]
anchor_head_mode = "excluded_classes"
ann_file_mode = "default"

# 95-5 percentile
point_cloud_range = [-22.54448366, -45.37521362, -2.45846081,
                     158.85762253,  42.21965866,  10.27910986]


anchor_head_sizes = [
        # catenary_pole
        # Coverage (IoU): 0.593|0.184|0.297|0.959|39(mean|std|min|max|uniques)
        [0.836, 0.86, 7.6568],
        [1.1333333333333333, 1.2, 8.299999999999999],
        # animal
        # Coverage (IoU): 0.481|0.228|0.119|1.000|10(mean|std|min|max|uniques)
        [0.45, 0.45, 0.4166666666666667],
        [0.9, 0.9, 0.5],
        [0.5666666666666667, 0.6333333333333333, 0.5666666666666667],
        # signal
        # Coverage (IoU): 0.650|0.244|0.300|1.000|7(mean|std|min|max|uniques)
        [0.6499999999999999, 0.6, 0.8500000000000001],
        [0.4, 1.0, 1.0],
        # person
        # Coverage (IoU): 0.489|0.242|0.085|0.933|93(mean|std|min|max|uniques)
        [0.7814285714285715, 0.8267857142857143, 1.872857142857143],
        [1.8, 1.56, 2.56],
        [0.9259259259259259, 0.9222222222222223, 2.364074074074074],
        # signal_pole
        # Coverage (IoU): 0.705|0.165|0.413|0.992|9(mean|std|min|max|uniques)
        [0.42500000000000004, 0.775, 6.425000000000001],
        [0.4, 0.5, 6.35],
        [0.5666666666666667, 0.7333333333333333, 6.4],
        # switch
        # Coverage (IoU): 0.567|0.254|0.118|0.988|96(mean|std|min|max|uniques)
        [22.305084745762713, 4.0288135593220336, 0.2],
        [7.12857142857143, 3.2904761904761903, 0.2],
        [14.837499999999999, 3.375, 0.2],
        # signal_bridge
        # Coverage (IoU): 1.000|0.000|1.000|1.000|1(mean|std|min|max|uniques)
        [2.5, 6.1, 0.7],
        # train
        # Coverage (IoU): 0.424|0.372|0.034|1.000|17(mean|std|min|max|uniques)
        [4.302, 3.3393333333333333, 3.512],
        [63.6, 4.1, 5.1],
        [11.27, 3.69, 3.82],
        # road_vehicle
        # Coverage (IoU): 0.358|0.341|0.020|0.970|44(mean|std|min|max|uniques)
        [5.1499999999999995, 3.06, 4.94],
        [3.269565217391304, 1.9260869565217393, 1.5782608695652174],
        [9.345454545454546, 4.418181818181818, 7.127272727272727],
        # bicycle
        # Coverage (IoU): 0.529|0.192|0.210|0.984|32(mean|std|min|max|uniques)
        [1.8785714285714286, 0.5, 1.3107142857142857],
        [1.690909090909091, 1.1454545454545455, 1.1636363636363636],
        [2.414285714285714, 0.5, 1.5142857142857142],
        # wagons
        # Coverage (IoU): 1.000|0.000|1.000|1.000|1(mean|std|min|max|uniques)
        [34.2, 2.6, 4.4],
        # buffer_stop
        # Coverage (IoU): 0.871|0.070|0.694|1.000|9(mean|std|min|max|uniques)
        [1.175, 2.775, 2.3],
        [1.1, 2.9, 2.5],
        [1.1, 2.55, 2.3499999999999996],
        # crowd
        # Coverage (IoU): 0.624|0.240|0.310|0.970|19(mean|std|min|max|uniques)
        [3.95, 1.15, 1.45],
        [4.009090909090909, 1.7454545454545454, 2.2],
        [3.5166666666666666, 1.1833333333333333, 1.4166666666666665],
    ]

ann_files = {
    "train": "osdar23_infos_train.pkl",
    "val": "osdar23_infos_val.pkl",
    "test": "osdar23_infos_test.pkl",
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
        with_label_3d=True,
    ),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[0.5, 0.5, 0.0],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816],
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.5, 0.5],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[1.0, 1.0, 0.5],
    ),
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
        keys=['points'],
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
        oversample_thr=0.3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=ann_files["train"],
            data_prefix=dict(pts='points'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR',
            metainfo=dict(classes=classes),
        ),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='points'),
        ann_file=ann_files["val"],
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
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='points'),
        ann_file=ann_files["test"],
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='lidar',
        metainfo=dict(classes=classes),
    ),
)


val_evaluator = dict(
    type='OSDaR23Metric',
    pklfile_prefix='results/osdar23/osdar23_results_train',
    result_prefix='results/osdar23/osdar23_metric_train',
)

test_evaluator = dict(
    type='OSDaR23Metric',
    pklfile_prefix='results/osdar23/osdar23_results_test',
    result_prefix='results/osdar23/osdar23_metric_test',
)
