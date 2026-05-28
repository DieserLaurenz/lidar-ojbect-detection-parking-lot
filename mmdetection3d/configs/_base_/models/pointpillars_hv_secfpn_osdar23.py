_base_ = ['../datasets/osdar23.py']

voxel_size = [0.3, 0.3, 1.5]
output_shape = [296, 608]
points_per_voxel = 37
max_voxels = (43000, 100000)

pillar_features = 128
anchor_features = [128, 128, 128]
second_channels = [64, 128, 256]
second_layer = [5, 7, 7]
second_layer_strides = [2, 2, 2]

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=points_per_voxel,
            point_cloud_range=_base_.point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=max_voxels,
        )
    ),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=_base_.point_cloud_dimension,
        feat_channels=(pillar_features,),
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=_base_.point_cloud_range),
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=pillar_features,
        output_shape=output_shape,
    ),
    backbone=dict(
        type='SECOND',
        in_channels=pillar_features,
        layer_nums=second_layer,
        layer_strides=second_layer_strides,
        out_channels=second_channels
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=second_channels,
        upsample_strides=[1, 2, 4],
        out_channels=anchor_features,
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=_base_.number_of_classes,
        in_channels=sum(anchor_features),
        feat_channels=sum(anchor_features),
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[_base_.point_cloud_range],
            size_per_range=True,
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
            loss_weight=2.0,
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
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.30,
                min_pos_iou=0.25,
                neg_iou_thr=0.05,
                ignore_iof_thr=-1)
        ] * len(_base_.anchor_head_sizes),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.3,
        score_thr=0.001,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=200),
)
