import os
from glob import glob
from PIL import Image
from typing import Callable, Optional
from functools import partial
from typing import Any, Tuple

import numpy as np
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import VisionDataset

import torchvision.transforms as transforms


class ImagenetDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img