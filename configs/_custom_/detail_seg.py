_base_ = '../segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py'
load_from = '/home/a3ilab01/treeai/mmsegmentation/work_dirs/segformer2/best_mIoU_iter_26000.pth'

# === Dataset ===
dataset_type = 'MyDataset1'
data_root = '/home/a3ilab01/treeai/dataset/segmentation/full/'

# ✅ Class 0 = background, excluded from training
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
    seg_pad_val=0,         
    size=img_scale
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
dict(
    type='RandomResize',
    scale=(640, 640),           
    ratio_range=(0.5, 2.0),     # scales from 320×320 to 1280×1280
    keep_ratio=True
),
    dict(type='RandomCrop', crop_size=(640, 640), cat_max_ratio=0.75),

    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.3, degree=10),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomCutOut', n_holes=1, cutout_shape=(32, 32), prob=0.1),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
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
                ignore_index=255,
            ),
            dict(type='DiceLoss', loss_weight=1.0, ignore_index=255)
        ]
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
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
    max_iters=40000,
    val_interval=1000
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
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
        patience=8,
        min_delta=0.0005,
        priority=75
    )
]

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=36000,
        eta_min=1e-6,
        by_epoch=False
    )
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
)
