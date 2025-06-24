import os.path as osp
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class MyDataset(BaseSegDataset):
    METAINFO = dict(
        classes=[f'class_{i}' for i in range(1, 62)],
        palette=[[i * 3 % 256, i * 7 % 256, i * 11 % 256] for i in range(1, 62)],
    )

    def load_data_list(self):
        data_list = []
        img_dir = osp.join(self.data_prefix['img_path'])
        seg_dir = osp.join(self.data_prefix['seg_map_path'])

        for img in sorted(self._scan_dir(img_dir)):
            seg = osp.join(seg_dir, osp.basename(img))
            if not osp.exists(seg):
                continue
            data_list.append(dict(
                img_path=img,
                seg_map_path=seg,
                seg_fields=['gt_seg_map'],           # ✅ NEW
                reduce_zero_label=False              # ✅ still required
            ))
        return data_list


    def _scan_dir(self, path):
        """Helper to list all .png files in a folder."""
        import os
        return [
            osp.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith('.png')
        ]
