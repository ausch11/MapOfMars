_base_ = [
    # '../_base_/models/mae_vit-base-p16.py',
    '../_base_/datasets/domars16k_bs512_mae.py',
    # '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]
train_dataloader = dict(batch_size=128, num_workers=4)
model = dict(
    type='MAE',
    backbone=dict(
        init_cfg=dict(type='Pretrained',
        checkpoint='/database/ygxiong/mmpretrain/pretrained/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth'),
        type='MAEViT', arch='b', patch_size=16, mask_ratio=0.75),
        neck=dict(
            type='MAEPretrainDecoder',
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.,
    ),
    head=dict(
        type='MAEPretrainHead',
        norm_pix=True,
        patch_size=16,
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    init_cfg=[
        dict(type='Xavier', layer='Linear', distribution='uniform'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])

# checkpoint =
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(
#         init_cfg=dict(
#             type='Pretrained', checkpoint=checkpoint, prefix='backbone')))


# val_dataloader = dict(batch_size=8, num_workers=5)
# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=500),
    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=20, max_keep_ckpts=20),
    sampler_seed = dict(type='DistSamplerSeedHook'),
    # validation results visualization, set True to enable it.
    visualization = dict(type='VisualizationHook', enable=False),
)

# val_evaluator = [
#   dict(type='Accuracy', topk=(1, 5)),
#   dict(type='AveragePrecision'),
#   dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
#   #dict(type='MultiLabelMetric', average='micro'),  # overall mean
# ]

randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=64)
