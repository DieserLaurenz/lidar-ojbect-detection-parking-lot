_base_ = ['./pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class-exp-merged-gtsample.py']

# Ablation "Unterboden-Filter": identisch zu merged-gtsample, aber die
# Eingabe-Punktwolken sind um Punkte unterhalb z = -2.0 bereinigt
# (Spiegelreflexions-Cluster 0.3-0.5 m unter dem Boden, siehe
# DATA_AUDIT.md §11 Kategorie 2). Boden liegt bei z ~ -1.74; entfernt
# wurden 0.09% aller Punkte (points_kitti_zfilt/, make_zfilt_bins).

train_dataloader = dict(
    dataset=dict(data_prefix=dict(pts='points_kitti_zfilt')))
val_dataloader = dict(
    dataset=dict(data_prefix=dict(pts='points_kitti_zfilt')))
test_dataloader = dict(
    dataset=dict(data_prefix=dict(pts='points_kitti_zfilt')))

val_evaluator = dict(
    pklfile_prefix='results/exp/merged_gtsample_zfilt_val_results',
    result_prefix='results/exp/merged_gtsample_zfilt_val_metric',
)
test_evaluator = dict(
    pklfile_prefix='results/exp/merged_gtsample_zfilt_test_results',
    result_prefix='results/exp/merged_gtsample_zfilt_test_metric',
)
