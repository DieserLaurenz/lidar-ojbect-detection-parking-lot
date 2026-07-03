_base_ = ['./pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class-exp.py']

train_dataloader = dict(dataset=dict(ann_file='exp_kitti_infos_merged_train.pkl'))
val_dataloader = dict(dataset=dict(ann_file='exp_kitti_infos_merged_val.pkl'))
test_dataloader = dict(dataset=dict(ann_file='exp_kitti_infos_merged_test.pkl'))

val_evaluator = dict(
    pklfile_prefix='results/exp/merged_val_results',
    result_prefix='results/exp/merged_val_metric',
)
test_evaluator = dict(
    pklfile_prefix='results/exp/merged_test_results',
    result_prefix='results/exp/merged_test_metric',
)
