_base_ = [
    '../swin/swin-large-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'  # noqa

# === Dataset ===
dataset_type = 'MyDataset'
data_root = '/home/a3ilab01/treeai/dataset/segmentation/full/'


classes = [f'class_{i}' for i in range(0, 62)]
palette = [[i * 3 % 256, i * 7 % 256, i * 11 % 256] for i in range(0, 62)]
metainfo = dict(classes=classes, palette=palette)


data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0,        
    size=(640, 640)
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', 
         scale=(640, 640),  # Resize the image to a fixed target size (can be any size you want)
         ratio_range=(0.5, 2.0)),  # Random aspect ratio range between 50% and 200%
    dict(type='RandomCrop', crop_size=(640, 640), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.3, degree=10),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomCutOut', n_holes=1, cutout_shape=(32, 32), prob=0.1),
    dict(type='PackSegInputs'),
]



test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640,640), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12),
    decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=62),
    auxiliary_head=dict(in_channels=768, num_classes=62))

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='annotations_0/train'),
        ann_file=None,
        pipeline=train_pipeline,
        metainfo=metainfo,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations_0/val'),
        ann_file=None,
        pipeline=test_pipeline,
        metainfo=metainfo,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=36000,
    val_interval=500
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=36000,
        eta_min=1e-6,
        by_epoch=False
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        max_keep_ckpts=2,
        save_best='mIoU',
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50)
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mIoU',
        rule='greater',
        patience=5,
        min_delta=0.0005,
        priority=75
    )
]

tta_pipeline = [
    dict(type='TestTimeAug', transforms=[
        [dict(type='Resize', scale=(768, 768), keep_ratio=True)],
        [dict(type='ResizeToMultiple', size_divisor=32)],
        [dict(type='RandomFlip', prob=1.0), dict(type='RandomFlip', prob=0.0)],
        [dict(type='LoadAnnotations', reduce_zero_label=False)],
        [dict(type='PackSegInputs')],
    ])
]


tta_model = dict(type='SegTTAModel')
  