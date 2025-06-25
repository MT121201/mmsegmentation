import os
import numpy as np
from mmseg.apis import init_model, inference_model
from mmengine.config import Config
from mmengine.registry import init_default_scope
from PIL import Image
import argparse
from tqdm import tqdm

def main(config_file, checkpoint_file, test_dir, output_dir, device='cuda:0'):
    # Load model
    cfg = Config.fromfile(config_file)
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    model = init_model(cfg, checkpoint_file, device=device)

    os.makedirs(output_dir, exist_ok=True)
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for fname in tqdm(test_images):
        img_path = os.path.join(test_dir, fname)
        result = inference_model(model, img_path)
        
        # Get predicted label map
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy()

        # Strip extension: '0001.png' → '0001'
        base_name = os.path.splitext(fname)[0]
        np.save(os.path.join(output_dir, base_name + '.npy'), pred.astype(np.uint8))

    print(f"\n✅ Saved {len(test_images)} .npy prediction files to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to model config (.py)')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--test_dir', required=True, help='Folder with test images (e.g., /path/images/test)')
    parser.add_argument('--output_dir', default='predictions', help='Where to save .npy files')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.test_dir, args.output_dir, args.device)
