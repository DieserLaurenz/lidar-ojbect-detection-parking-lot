_base_ = [
    # model includes dataset file!
    '../_base_/models/pointpillars_hv_secfpn_osdar23.py',
    '../_base_/default_runtime.py',
]

vis_backends = [
    dict(
        type='LocalVisBackend',
        save_dir="/data/lumpi/output",
    ),
]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

custom_hooks = [
    dict(type='StopOnNaNHook'),
]

max_epochs = 30

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=4,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
lr = 1e-3

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.001,
    ),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=lr,
        total_steps=max_epochs,
        anneal_strategy='cos',
    ),

]

auto_scale_lr = dict(enable=True, base_batch_size=8*_base_.batch_size)


resume = False
load_from = None
