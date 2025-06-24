from mmseg.datasets.basesegdataset import CustomDataset
from mmseg.registry import DATASETS
print('CustomDataset' in DATASETS.module_dict)  # Should print True
