_base_ = ["./pointpillars_hv_secfpn_sbn-all_8xb4-2x_osdar23.py"]

# Override OSDaR23 Settings for Testing Experiment Data!

ann_files = {
    "merged": "exp_osdar_infos_merged.pkl",
    "os0": "exp_osdar_infos_os0.pkl",
    "os1": "exp_osdar_infos_os1.pkl",
}

ann_file = ann_files['merged']

test_dataloader = dict(
    dataset=dict(
        type="LUMPIDataset",
        data_root='data/exp',
        ann_file=ann_file,
        data_prefix=dict(pts='points_osdar'),
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        box_type_3d='lidar',
    ),
)


test_evaluator = dict(
    pklfile_prefix='results/exp/exp_osdar23_results_test',
    result_prefix='results/exp/exp_osdar23_metric_test',
    score_threshold=0.0,
)

model = dict(
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.1,
        score_thr=0.2,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=54,
    ),
)

custom_hooks = [
    dict(type='InferenceTimeHook'),
]
