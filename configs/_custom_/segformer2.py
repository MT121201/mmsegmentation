_base_ = '../segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py'

# === Dataset ===
dataset_type = 'MyDataset'
data_root = '/home/a3ilab01/treeai/dataset/segmentation/full/'

classes = [f'class_{i}' for i in range(1, 62)]
palette = [[i * 3 % 256, i * 7 % 256, i * 11 % 256] for i in range(1, 62)]
metainfo = dict(classes=classes, palette=palette)


img_scale = (768, 768)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,  # Must match ignore_index
    size=(768, 768)
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.3, degree=10),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomCutOut', n_holes=1, cutout_shape=(32, 32), prob=0.1),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs'),
]

model = dict(
    decode_head=dict(
        num_classes=61,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                ignore_index=255,  # <== match seg_pad_val
            ),
            dict(type='DiceLoss', loss_weight=1.0, ignore_index=255)
        ]
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,            # Stage 3 output of mit-b5
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=61,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, ignore_index=255),
            dict(type='DiceLoss', loss_weight=0.4, ignore_index=255),
        ]
    ),

)


train_dataloader = dict(
    batch_size=2,
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
    max_iters=36000,
    val_interval=2000
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
        interval=2000,
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
        [dict(type='Resize', scale=img_scale, keep_ratio=True)],
        [dict(type='RandomFlip', prob=1.0), dict(type='RandomFlip', prob=0.0)],
        [dict(type='LoadAnnotations', reduce_zero_label=True)],
        [dict(type='PackSegInputs')],
    ])
]

tta_model = dict(type='SegTTAModel')
  