_base_ = ['../datasets/lumpi.py']

voxel_size = [0.3, 0.3, 4]
point_per_voxel = 25

pillar_features = 128
anchor_features = 128
second_channels = [64, 128, 256]

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=point_per_voxel,
            point_cloud_range=_base_.point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=50000,
        )
    ),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[pillar_features],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=_base_.point_cloud_range),
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=pillar_features,
        output_shape=[328, 320],
    ),
    backbone=dict(
        type='SECOND',
        in_channels=pillar_features,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=second_channels),
    neck=dict(
        type='SECONDFPN',
        in_channels=second_channels,
        upsample_strides=[1, 2, 4],
        out_channels=[anchor_features] * 3
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=_base_.number_of_classes,
        in_channels=anchor_features * 3,
        feat_channels=anchor_features * 3,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                _base_.point_cloud_range
            ],
            sizes=_base_.anchor_head_sizes,
            reshape_out=False,
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0,
            loss_weight=3.0,
        ),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # person
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.20,
                neg_iou_thr=0.08,
                min_pos_iou=0.18,
                ignore_iof_thr=-1),
            dict(  # car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.28,
                neg_iou_thr=0.10,
                min_pos_iou=0.18,
                ignore_iof_thr=-1),
            dict(  # bicycle
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.3,
                neg_iou_thr=0.10,
                min_pos_iou=0.20,
                ignore_iof_thr=-1),
            dict(  # motorcycle
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.3,
                neg_iou_thr=0.10,
                min_pos_iou=0.20,
                ignore_iof_thr=-1),
            dict(  # bus
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.3,
                neg_iou_thr=0.10,
                min_pos_iou=0.20,
                ignore_iof_thr=-1),
            dict(  # truck
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.3,
                neg_iou_thr=0.10,
                min_pos_iou=0.20,
                ignore_iof_thr=-1),
            dict(  # unknown
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.2,
                neg_iou_thr=0.05,
                min_pos_iou=0.1,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=True,
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.30,
        score_thr=0.20,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=200,
    ),
)
