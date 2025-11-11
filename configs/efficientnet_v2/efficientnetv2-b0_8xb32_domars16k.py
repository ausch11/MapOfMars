_base_ = [
    '../_base_/models/efficientnet_v2/efficientnetv2_b0.py',
    # '../_base_/datasets/imagenet_bs32.py',
    '../_base_/datasets/domars16k_bs128_poolformer_small_224.py',
#    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]


checkpoint = r'D:\ygxiong\Code\mmpretrain\pretrained\efficientnetv2-b0_3rdparty_in1k_20221221-9ef6e736.pth'
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

#
# # dataset settings
# dataset_type = 'ImageNet'
# data_preprocessor = dict(
#     num_classes=1000,
#     # RGB format normalization parameters
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     # convert image from BGR to RGB
#     to_rgb=True,
# )
#
# bgr_mean = data_preprocessor['mean'][::-1]
# bgr_std = data_preprocessor['std'][::-1]
#
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='RandomResizedCrop',
#         scale=192,
#         backend='pillow',
#         interpolation='bicubic'),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(
#         type='RandAugment',
#         policies='timm_increasing',
#         num_policies=2,
#         total_level=10,
#         magnitude_level=9,
#         magnitude_std=0.5,
#         hparams=dict(
#             pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
#     dict(
#         type='RandomErasing',
#         erase_prob=0.25,
#         mode='rand',
#         min_area_ratio=0.02,
#         max_area_ratio=1 / 3,
#         fill_color=bgr_mean,
#         fill_std=bgr_std),
#     dict(type='PackInputs'),
# ]
#
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='EfficientNetCenterCrop', crop_size=224, crop_padding=0),
#     dict(type='PackInputs'),
# ]
#
# train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
# val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
# test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
