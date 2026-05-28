_base_ = [
    'pointpillars_hv_secfpn_sbn-all_8xb4-2x_lumpi.py',
]

# Override LUMPI Settings for Testing Experiment Data!

ann_files = {
    "merged": "exp_lumpi_infos_merged.pkl",
    "os0": "exp_lumpi_infos_os0.pkl",
    "os1": "exp_lumpi_infos_os1.pkl",
}

ann_file = ann_files['merged']


test_dataloader = dict(
    dataset=dict(
        data_root='data/exp',
        data_prefix=dict(pts='points_lumpi'),
        ann_file=ann_file,
    )
)

test_evaluator = dict(
    pklfile_prefix='results/exp/exp_lumpi_results_test',
    result_prefix='results/exp/exp_lumpi_metric_test',
    score_threshold=0.0,
)

model = dict(
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.35,
        score_thr=0.3,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=30,
    ),
)

custom_hooks = [
    dict(type='InferenceTimeHook'),
]
