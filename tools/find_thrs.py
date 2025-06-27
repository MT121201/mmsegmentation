import numpy as np
from mmseg.apis import init_model, inference_model
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.evaluation import IoUMetric
import torch
import os
from tqdm import tqdm

# Load model
config_file = '/home/a3ilab01/treeai/mmsegmentation/configs/_custom_/segformer2.py'
checkpoint_file = '/home/a3ilab01/treeai/det_tree/weights/seg_best52.pth'
cfg = Config.fromfile(config_file)
init_default_scope(cfg.get('default_scope', 'mmseg'))
model = init_model(cfg, checkpoint_file, device='cuda:0')
model.eval()

# Get val dataloader
val_loader = cfg.val_dataloader
from mmseg.datasets import build_dataset
from mmseg.datasets.utils import build_dataloader

val_dataset = build_dataset(cfg.val_dataloader['dataset'])
data_loader = build_dataloader(
    val_dataset,
    samples_per_gpu=cfg.val_dataloader.get('batch_size', 1),
    workers_per_gpu=cfg.val_dataloader.get('num_workers', 4),
    dist=False,
    shuffle=False
)



# Define thresholds to test
thresholds = np.linspace(0.1, 0.95, 18)
results_per_thresh = []

# Evaluation loop
for thresh in thresholds:
    print(f"\n[Threshold = {thresh:.2f}]")
    preds, gts = [], []
    
    for data in tqdm(data_loader):
        with torch.no_grad():
            output = model.test_step(data)[0]
            logits = output['seg_logits'].squeeze(0)  # shape: [C, H, W]
            probs = torch.softmax(logits, dim=0)      # softmax over channel dim
            max_probs, pred = probs.max(dim=0)        # shape: [H, W]

            pred = pred + 1                           # shift back label indices (0→1, ..., 60→61)
            pred[max_probs < thresh] = 0              # set as background

            preds.append(pred.cpu().numpy().astype(np.uint8))
            gt = data['data_samples'][0].gt_sem_seg.data.squeeze().cpu().numpy()
            gts.append(gt)

    # Evaluate IoU
    evaluator = IoUMetric(iou_metrics=['mIoU'], ignore_index=None, num_classes=62)
    evaluator.process(preds, gts)
    metrics = evaluator.evaluate()
    mIoU = metrics['mIoU']
    print(f"mIoU @ {thresh:.2f} = {mIoU:.4f}")
    results_per_thresh.append((thresh, mIoU))

# Find best threshold
best_thresh, best_iou = max(results_per_thresh, key=lambda x: x[1])
print(f"\n✅ Best Threshold: {best_thresh:.2f} with mIoU: {best_iou:.4f}")
