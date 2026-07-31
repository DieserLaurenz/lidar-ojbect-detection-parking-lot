_base_ = ['./centerpoint_pillar016_second_secfpn_8xb6-50e_exp-3d.py']

# os1-View; eigener Ergebnis-Prefix, damit parallele Validierungen
# (merged-Lauf auf GPU 0) sich nicht die Dump-Dateien überschreiben.

train_dataloader = dict(
    dataset=dict(ann_file='exp_kitti_infos_os1_train.pkl'))
val_dataloader = dict(
    dataset=dict(ann_file='exp_kitti_infos_os1_val.pkl'))
test_dataloader = dict(
    dataset=dict(ann_file='exp_kitti_infos_os1_test.pkl'))

val_evaluator = dict(
    pklfile_prefix='results/exp_cp_os1/exp_kitti_results_val',
    result_prefix='results/exp_cp_os1/exp_kitti_metric_val')
test_evaluator = dict(
    pklfile_prefix='results/exp_cp_os1/exp_kitti_results_test',
    result_prefix='results/exp_cp_os1/exp_kitti_metric_test')
