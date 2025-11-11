_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/domars16k_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))

checkpoint = '/database/ygxiong/mmpretrain/pretrained/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    head=dict(num_classes=15))
batch_size = 32
train_dataloader = dict(batch_size=batch_size, num_workers=5)

val_dataloader = dict(batch_size=batch_size, num_workers=5)

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=2)
auto_scale_lr = dict(base_batch_size=4*batch_size)
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=500),
    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='auto'),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# schedule settings
val_evaluator = [
    dict(type='Accuracy', topk=(1, 5)),
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric', average='micro'),  # overall mean
    dict(type='MultiLabelMetric', average=None),  # class-wise mean
]

test_evaluator = val_evaluator