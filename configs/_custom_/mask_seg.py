_base_ = '../segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py'

# === Dataset ===
dataset_type = 'MyDataset'
data_root = '/home/a3ilab01/treeai/dataset/segmentation/full/'

classes = ['background', 'foreground']
palette = [[0, 0, 0], [255, 255, 255]]  # background and foreground color palette
metainfo = dict(classes=classes, palette=palette)

img_scale = (640, 640)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,  # background class as 255, will be ignored
    size=(640, 640)
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),  # Let model learn background and foreground
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
    dict(type='LoadAnnotations', reduce_zero_label=False),  # Keep background learning
    dict(type='PackSegInputs'),
]

# Model configuration
model = dict(
    decode_head=dict(
        num_classes=2,  # Background + foreground
        loss_decode=dict(
            type='CrossEntropyLoss',  # Switching to Cross-Entropy Loss for better binary segmentation
            use_sigmoid=True,  # Binary segmentation uses sigmoid, not softmax
            loss_weight=1.0,
            # ignore_index=255  # Ignore background during loss calculation
        )
    ),
    auxiliary_head=None  # optional, can re-add if you want
)

# Data loading configuration
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='bin_mask/train'),
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
        data_prefix=dict(img_path='images/val', seg_map_path='bin_mask/val'),
        ann_file=None,
        pipeline=test_pipeline,
        metainfo=metainfo,
    )
)

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# Training configuration
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=36000,
    val_interval=100  # Validate after every 100 iterations
)

# Parameter scheduler
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=36000,
        eta_min=1e-6,
        by_epoch=False
    )
]

# Default hooks for logging and checkpoints
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

# Early stopping hook for monitoring overfitting
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
