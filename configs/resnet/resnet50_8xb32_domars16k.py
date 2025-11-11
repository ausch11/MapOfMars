_base_ = [
    '../_base_/models/resnet50.py',
   # '../_base_/datasets/imagenet_bs32.py',
    '../_base_/datasets/domars16k_bs128_poolformer_small_224.py',

    # '../_base_/schedules/imagenet_bs256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',

    '../_base_/default_runtime.py'
]

checkpoint = r'D:\ygxiong\Code\mmpretrain\pretrained\resnet50_8xb32_in1k_20210831-ea4938fc.pth'
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