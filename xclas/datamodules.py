from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize, normalize
from torchvision.io import read_image, ImageReadMode as IRM

from typing import Optional, Any, Tuple, List, Union
from pathlib import Path
import torch

_mean = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]


def line_split(line: str, root: str) -> Tuple[str, int]:
    root = root or ''
    x = line.split('\t')
    return str(Path(root, x[0])), int(x[1])


class FileDataset(Dataset):
    def __init__(self, file_or_dir: str, root: str = None):
        super().__init__()

        if root is None:  # image file or folder
            file_or_dir = Path(file_or_dir)
            if file_or_dir.is_file():
                self.dataset = [(file_or_dir, 0)]
            else:
                self.dataset = [
                    (str(fpath), 0) for fpath in file_or_dir.rglob("*.*")
                    if fpath.suffix.lower() in {'.jpg', '.png', '.jpeg'}
                ]
        else:
            lines = Path(root, file_or_dir).read_text().splitlines()
            self.dataset = [line_split(line, root) for line in lines]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[str, int]:
        fpath, target = self.dataset[idx]
        return fpath, target


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset: FileDataset,
        img_size: tuple = None,
    ):
        if img_size is None:
            img_size = (224, 224)
        super().__init__()

        self.dataset = dataset
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[torch.tensor, int]:
        '''[ch, H, W], target:int'''
        # fpath, target = super().dataset[index]
        fpath, target = self.dataset[index]

        img = read_image(fpath, IRM.RGB)  # [RGB, H, W]
        x = resize(img, self.img_size).float() / 255.
        x = normalize(x, _mean, _std)
        return x, target


def train_dataloader(
    data_dir,
    fileset,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: List[int] = None
) -> DataLoader:
    fileset = FileDataset(fileset, data_dir)
    dataset = ImageDataset(fileset, img_size)
    ret = DataLoader(
        dataset,
        batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True
    )
    return ret


def valid_dataloader(
    data_dir,
    fileset,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: List[int] = None
) -> DataLoader:
    fileset = FileDataset(fileset, data_dir)
    dataset = ImageDataset(fileset, img_size)
    ret = DataLoader(
        dataset,
        batch_size,
        num_workers = num_workers,
        shuffle = False,
        pin_memory = False
    )
    return ret


def predict_dataloader(
    root,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: List[int] = None,
) -> DataLoader:
    fileset = FileDataset(root)
    dataset = ImageDataset(fileset, img_size)
    return DataLoader(
        dataset,
        batch_size,
        num_workers = num_workers,
        pin_memory = False,
        shuffle = False
    )
