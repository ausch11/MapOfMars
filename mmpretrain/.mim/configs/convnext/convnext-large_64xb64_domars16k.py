_base_ = [
    '../_base_/models/convnext/convnext-large.py',
    '../_base_/datasets/domars16k_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# dataset setting
train_dataloader = dict(batch_size=32)
checkpoint = '/database/ygxiong/mmpretrain/pretrained/convnext-large_3rdparty_in21k_20220124-41b5a79f.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.')),
)

    # schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=4e-3),
    clip_grad=None,
)
val_evaluator = [
    dict(type='Accuracy', topk=(1, 5)),
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric', average='micro'),  # overall mean
    dict(type='MultiLabelMetric', average=None)  # class-wise mean
]

train_cfg = dict(by_epoch=True, max_epochs=300)
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='auto'))
# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (64 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=4096)
