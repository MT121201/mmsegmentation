import os
import numpy as np
from tqdm import tqdm
from mmseg.apis import inference_model, init_model
from mmengine.config import Config
from mmengine.registry import init_default_scope
import torch

# --- Paths ---
config_path = '/home/a3ilab01/treeai/mmsegmentation/configs/_custom_/segformer2.py'
checkpoint_path = '/home/a3ilab01/treeai/mmsegmentation/work_dirs/segformer2/best_mIoU_iter_26000.pth'
img_dir = '/home/a3ilab01/treeai/dataset/SemSeg_test-images'
output_dir = './predictions'

os.makedirs(output_dir, exist_ok=True)

# --- Load model ---
cfg = Config.fromfile(config_path)
init_default_scope(cfg.get('default_scope', 'mmseg'))
model = init_model(cfg, checkpoint_path, device='cuda:0')
model.eval()

# --- Inference loop ---
test_images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

for fname in tqdm(test_images):
    img_path = os.path.join(img_dir, fname)

    # Run inference and get logits
    with torch.no_grad():
        result = inference_model(model, img_path)
        pred = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)  # no shift

    base_name = os.path.splitext(fname)[0]
    np.save(os.path.join(output_dir, base_name + '.npy'), pred)

print(f"✅ Raw predictions saved to: {output_dir}")
