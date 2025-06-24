_base_ = '../segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py'

dataset_type = 'MyDataset'
data_root = '/home/a3ilab01/treeai/dataset/segmentation/full/'

classes = [f'class_{i}' for i in range(1, 62)]  # 1 to 61
palette = [[i * 3 % 256, i * 7 % 256, i * 11 % 256] for i in range(1, 62)]
metainfo = dict(classes=classes, palette=palette)


img_scale = (640, 640)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0,         # ✅ ignore class ID 0
    size=(640, 640)
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
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
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
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
    max_iters=40000,
    val_interval=2000
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,        # ✅ Match val_interval
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
