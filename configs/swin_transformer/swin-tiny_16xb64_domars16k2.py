_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/domars16k_bs128_poolformer_small_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))

checkpoint = 'D:\ygxiong\Code\mmpretrain\pretrained\swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    )
batch_size = 128
train_dataloader = dict(batch_size=batch_size)

default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=5, save_best='auto'))
val_evaluator = [
    dict(type='MultiLabelMetric', average='micro'),  # overall mean
    dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric', average=None),  # class-wise mean
    dict(type='Accuracy', topk=(1, 5)),
    dict(type='AveragePrecision'),
]
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=5)

auto_scale_lr = dict(base_batch_size=1024)
