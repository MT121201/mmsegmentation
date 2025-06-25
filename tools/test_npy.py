from mmseg.apis import init_model, inference_model
from mmseg.models import SegTTAModel  # 👈 required for TTA
from mmengine.config import Config
from mmengine.registry import init_default_scope
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import torch

def main(config_file, checkpoint_file, test_dir, output_dir, device='cuda:0'):
    # Load config and init model
    cfg = Config.fromfile(config_file)
    init_default_scope(cfg.get('default_scope', 'mmseg'))

    # ✅ Load base model
    model = init_model(cfg, checkpoint_file, device=device)

    # ✅ Wrap with TTA model if defined in config
    if 'tta_model' in cfg and 'tta_pipeline' in cfg:
        model = SegTTAModel(**cfg.tta_model)
        model.cfg = cfg  # this is required for model to have correct pipeline
        model.base_model = init_model(cfg, checkpoint_file, device=device)
        model.base_model.eval()
        model.eval()

    os.makedirs(output_dir, exist_ok=True)
    test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for fname in tqdm(test_images):
        img_path = os.path.join(test_dir, fname)

        # Use TTA if available
        result = inference_model(model, img_path)

        pred = result.pred_sem_seg.data.squeeze().cpu().numpy()
        base_name = os.path.splitext(fname)[0]
        np.save(os.path.join(output_dir, base_name + '.npy'), pred.astype(np.uint8))

    print(f"\n✅ Saved {len(test_images)} .npy prediction files to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to model config (.py)')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--test_dir', required=True, help='Folder with test images')
    parser.add_argument('--output_dir', default='predictions', help='Output folder for .npy files')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.test_dir, args.output_dir, args.device)
