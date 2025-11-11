_base_ = [
    '../_base_/models/poolformer/poolformer_s36.py',
    '../_base_/datasets/domars16k_bs128_poolformer_small_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]
train_dataloader = dict(batch_size=64)
checkpoint = '/database/ygxiong/mmpretrain/pretrained/poolformer-s36_3rdparty_32xb128_in1k_20220414-d78ff3e8.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.', _delete_=True)))
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

# schedule settings
optim_wrapper = dict(
    optimizer=dict(lr=4e-3),
    clip_grad=dict(max_norm=5.0),
)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
# auto_scale_lr = dict(base_batch_size=4096)
