_base_ = [
    '../_base_/models/mvit/mvitv2-base.py',
    '../_base_/datasets/domars16k_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]
checkpoint = 'D:\ygxiong\Code\mmpretrain\pretrained\mvitv2-base_3rdparty_in1k_20220722-9c4f0a17.pth'
model = dict(
    backbone=dict(
        type='MViT_mars2', arch='base', drop_path_rate=0.3,
        #type='MViT', arch='base', drop_path_rate=0.3,

        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.')))

# dataset settings
train_dataloader = dict(batch_size=64)
val_dataloader = dict(batch_size=64)
test_dataloader = dict(batch_size=64)

default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=50, save_best='auto'))
val_evaluator = [
    dict(type='MultiLabelMetric', average='micro'),  # overall mean
    dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric', average=None),  # class-wise mean
    dict(type='Accuracy', topk=(1, 5)),
    dict(type='AveragePrecision'),
]
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=5)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(lr=2.5e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.pos_embed': dict(decay_mult=0.0),
            '.rel_pos_h': dict(decay_mult=0.0),
            '.rel_pos_w': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=1.0),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=70,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=70)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)
