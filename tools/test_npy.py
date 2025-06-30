import os
import urllib.request
import numpy as np
from tqdm import tqdm
import torch
from argparse import ArgumentParser

from mmseg.apis import inference_model, init_model
from mmengine.config import Config
from mmengine.registry import init_default_scope

# === Paths ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(ROOT_DIR, 'configs', '_custom_', 'segformer2.py')
WEIGHTS_DIR = 'weights'
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, 'seg_37.pth')
CHECKPOINT_URL = 'https://huggingface.co/datasets/MinTR-KIEU/det_tree/resolve/main/weights/seg_37.pth'
OUTPUT_DIR = os.path.join(ROOT_DIR, 'predictions')
DEVICE = 'cuda:0'

def ensure_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n🔍 Checkpoint not found at {CHECKPOINT_PATH}")
        print(f"⬇️  Downloading from: {CHECKPOINT_URL}")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        try:
            urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
            print("✅ Download complete.\n")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download checkpoint: {e}")

def main(test_dir):
    ensure_checkpoint()

    cfg = Config.fromfile(CONFIG_PATH)
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    model = init_model(cfg, CHECKPOINT_PATH, device=DEVICE)
    model.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_images = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    for fname in tqdm(test_images, desc='Running SegFormer inference'):
        img_path = os.path.join(test_dir, fname)

        with torch.no_grad():
            result = inference_model(model, img_path)
            pred = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)

        base_name = os.path.splitext(fname)[0]
        np.save(os.path.join(OUTPUT_DIR, base_name + '.npy'), pred)

    print(f"\n✅ Raw predictions saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True, help='Directory of test images')
    args = parser.parse_args()
    main(args.test_dir)
