import os
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from mmseg.apis import inference_model, init_model
from mmengine.config import Config
from mmengine.registry import init_default_scope

# === Model A: background detector ===
cfg_A = Config.fromfile('/home/a3ilab01/treeai/mmsegmentation/configs/_custom_/mask_seg.py')
init_default_scope(cfg_A.get('default_scope', 'mmseg'))
model_A = init_model(cfg_A, '/home/a3ilab01/treeai/mmsegmentation/work_dirs/mask_seg/best_mIoU_iter_300.pth', device='cuda:0')
model_A.eval()

# === Model B: detailed segmenter ===
cfg_B = Config.fromfile('/home/a3ilab01/treeai/mmsegmentation/configs/_custom_/detail_seg.py')
init_default_scope(cfg_B.get('default_scope', 'mmseg'))
model_B = init_model(cfg_B, '/home/a3ilab01/treeai/det_tree/weights/seg_best52.pth', device='cuda:0')
model_B.eval()

# === Paths ===
img_dir = '/home/a3ilab01/treeai/dataset/SemSeg_test-images'
output_dir = './predictions'
os.makedirs(output_dir, exist_ok=True)

# === Loop over images ===
test_images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

for fname in tqdm(test_images):
    img_path = os.path.join(img_dir, fname)

    # === Step 1: Predict BG with Model A ===
    with torch.no_grad():
        result_A = inference_model(model_A, img_path)
        pred_A = result_A.pred_sem_seg.data.squeeze(0)  # [H, W]
        bg_mask = (pred_A == 0).cpu().numpy()  # 1 where BG, 0 elsewhere

    # === Step 2: Mask image (set BG to 255 or black) ===
    img_orig = Image.open(img_path).convert('RGB')
    img_np = np.array(img_orig)
    img_np[bg_mask] = 255  # or [0,0,0] for black

    # Save masked image temporarily
    masked_img_path = os.path.join(output_dir, 'temp_masked.png')
    Image.fromarray(img_np).save(masked_img_path)

    # === Step 3: Run Model B on masked image ===
    with torch.no_grad():
        result_B = inference_model(model_B, masked_img_path)
        pred_B = result_B.pred_sem_seg.data.squeeze(0)  # [H, W]
        pred_B = pred_B + 1  # shift to [1, N]

    # === Step 4: Fuse final prediction ===
    pred_fused = pred_B.clone()
    pred_fused[pred_A == 0] = 0  # insert BG from Model A

    # === Step 5: Save output ===
    base_name = os.path.splitext(fname)[0]
    np.save(os.path.join(output_dir, base_name + '.npy'), pred_fused.cpu().numpy().astype(np.uint8))

# Cleanup temp
if os.path.exists(os.path.join(output_dir, 'temp_masked.png')):
    os.remove(os.path.join(output_dir, 'temp_masked.png'))

print(f"✅ Masked-fusion predictions saved to: {output_dir}")
