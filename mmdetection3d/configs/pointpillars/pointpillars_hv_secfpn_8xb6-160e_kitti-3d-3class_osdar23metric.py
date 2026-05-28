_base_ = ['./pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py']

# Override KITTI defaults to use OSDaR23 Metric

test_evaluator = dict(
    _delete_=True,
    type='OSDaR23Metric',
    pklfile_prefix='results/kitti/kitti_osdar23metric_results_test',
    result_prefix='results/kitti/kitti_osdar23metric_metric_test',
    score_threshold=0.0,
)
