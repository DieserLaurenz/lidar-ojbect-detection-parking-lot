_base_ = ['./pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class-exp.py']

train_dataloader = dict(dataset=dict(ann_file='exp_kitti_infos_os0_train.pkl'))
val_dataloader = dict(dataset=dict(ann_file='exp_kitti_infos_os0_val.pkl'))
test_dataloader = dict(dataset=dict(ann_file='exp_kitti_infos_os0_test.pkl'))

val_evaluator = dict(
    pklfile_prefix='results/exp/os0_val_results',
    result_prefix='results/exp/os0_val_metric',
)
test_evaluator = dict(
    pklfile_prefix='results/exp/os0_test_results',
    result_prefix='results/exp/os0_test_metric',
)
