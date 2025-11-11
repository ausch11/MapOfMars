_base_ = [
    '../_base_/models/inception_v3.py',
    #'../_base_/datasets/domars16k_bs32.py',
    '../_base_/datasets/domars16k_bs128_poolformer_small_224.py',

    # '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]
checkpoint = 'D:\ygxiong\Code\mmpretrain\pretrained\inception-v3_3rdparty_8xb32_in1k_20220615-dcd4d910.pth'
train_dataloader = dict(batch_size=64)

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.')),)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomResizedCrop', scale=299),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(type='PackInputs'),
# ]
#
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='ResizeEdge', scale=342, edge='short'),
#     dict(type='CenterCrop', crop_size=299),
#     dict(type='PackInputs'),
# ]

val_evaluator = [
    dict(type='MultiLabelMetric', average='micro'),  # overall mean
    dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric', average=None),  # class-wise mean
    dict(type='Accuracy', topk=(1, 5)),
    dict(type='AveragePrecision'),
]
# train_dataloader = dict(batch_size=64, dataset=dict(pipeline=train_pipeline))
# val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
# test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=5, save_best='auto'))
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=5)