_base_ = [
    '../_base_/models/convnext_v2/tiny.py',
    '../_base_/datasets/domars16k_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]
checkpoint = '/database/ygxiong/mmpretrain/pretrained/convnext-v2-tiny_3rdparty-fcmae_in1k_20230104-80513adc.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.')))

# dataset setting
train_dataloader = dict(batch_size=64)   # 32=9K

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=3.2e-3),
    clip_grad=None,
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=40,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=40)
]
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=5, save_best='auto'))
val_evaluator = [
    dict(type='Accuracy', topk=(1, 5)),
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric', average='micro'),  # overall mean
    dict(type='MultiLabelMetric', average=None)  # class-wise mean
]
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=2)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]
