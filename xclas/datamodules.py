from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms.functional import resize, normalize
from torchvision import transforms as transes
from torchvision.io import read_image, ImageReadMode as IRM

from typing import Optional, Any, Tuple, List, Union
from pathlib import Path
import torch

_mean = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]



class FileDataset(Dataset):
    def __init__(self, file_or_dir: str, root: str = None, target_is_int:bool = True):
        super().__init__()
        self.root = root
        self.target_is_int = target_is_int

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
            a, b = zip(*[self.line_split(line) for line in lines])
            self.files, self.targets = list(a), list(b)            

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx) -> Tuple[str, int]:
        fpath, target = self.files[idx], self.targets[idx]
        return str(Path(self.root, fpath)), target
    
    def line_split(self,line: str) -> Tuple[str, int]:
        x = line.split('\t')
        target = torch.tensor([int(i) if self.target_is_int else float(i) for i in x[1:]])
        if len(target) == 1:
            target = target[0]
        return x[0], target

    def weights(self)->list[float]:
        # targets = [x[1] for x in self.dataset]
        _, idxes, counts = torch.unique(
            torch.tensor(self.targets),
            return_counts = True,
            return_inverse = True
        )
        weights = len(self.targets) / counts[idxes]
        return weights


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset: FileDataset,
        img_size: tuple = None,
        use_aug: bool = False,
    ):
        if img_size is None:
            img_size = (224, 224)
        super().__init__()

        self.dataset = dataset
        self.img_size = img_size
        self.trans = transes.Compose([  
            transes.Resize(size=img_size)
         ])
        if use_aug:
            self.trans = transes.Compose([
                transes.autoaugment.TrivialAugmentWide(),
                # transes.autoaugment.AugMix(),
                transes.RandomHorizontalFlip(0.5),
                transes.RandomResizedCrop(
                    size = img_size, scale = (0.8, 1.)
                ),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[torch.tensor, any]:
        '''[ch, H, W], target:int'''
        fpath, target = self.dataset[index]

        img = read_image(fpath, IRM.RGB)  # [RGB, H, W]
        img = self.trans(img)
        x = img / 255.
        x = normalize(x, _mean, _std)
        return x, target


def train_dataloader(
    data_dir,
    fileset,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: List[int] = None,
    use_aug: bool = True
) -> DataLoader:
    fileset = FileDataset(fileset, data_dir)
    dataset = ImageDataset(fileset, img_size, use_aug = use_aug)
    ret = DataLoader(
        dataset,
        batch_size,
        sampler = WeightedRandomSampler(fileset.weights(), len(fileset)),
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
