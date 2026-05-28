_base_ = [
    './pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py',
]

# Overrite KITTI defaults to use experiment data

ann_files = {
    "merged": "exp_kitti_infos_merged.pkl",
    "os0": "exp_kitti_infos_os0.pkl",
    "os1": "exp_kitti_infos_os1.pkl",
}
ann_file = ann_files["merged"]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
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

test_dataloader = dict(
    dataset=dict(
        type="LUMPIDataset",
        metainfo=dict(classes=['Pedestrian', 'Cyclist', 'Car']),
        data_root='data/exp',
        ann_file=ann_file,
        pipeline=test_pipeline,
        data_prefix=dict(pts='points_kitti'),
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        box_type_3d='lidar',
    ),
)
test_evaluator = dict(
    _delete_=True,
    type='OSDaR23Metric',
    pklfile_prefix='results/exp/exp_kitti_results_test',
    result_prefix='results/exp/exp_kitti_metric_test',
    collect_device='gpu',
    dump_only=False,
    score_threshold=0.0,
)
