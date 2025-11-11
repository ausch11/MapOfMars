_base_ = [
    '../_base_/datasets/domars16k_rsmamba.py',
    '../_base_/schedules/uc_schedule.py',
    '../_base_/rsmamba_default_runtime.py',
    #'../_base_/default_runtime.py',

]
train_dataloader = dict(batch_size=32)
train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=5)

num_classes = 15
data_preprocessor = dict(
    num_classes=num_classes,
)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RSMamba',
        arch='b',
        pe_type='learnable',
        path_type='forward_reverse_shuffle_gate',
        cls_position='none',  # 'head', 'tail', 'head_tail', 'middle', 'none'
        out_type='avg_featmap',
        img_size=224,
        patch_size=16,
        drop_rate=0.,
        patch_cfg=dict(stride=8),
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=192,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)


