from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, BatchSampler
from torchvision import transforms
from PIL import Image as PILImage
from accimage import Image as ACCImage

from typing import Optional, Any, Tuple
from pathlib import Path


def image_open(path: str) -> Any:
    try:
        return ACCImage(path)
    except IOError:
        return PILImage.open(path).convert('RGB')


def line_split(line: str) -> Tuple[str, int]:
    x = line.split('\t')
    return x[0], int(x[1])


class PredictImageset(Dataset):
    def __init__(self, file_or_dir: str, transform: Optional[transforms.Compose] ):
        super().__init__()
        self.trans = transform

        file_or_dir = Path(file_or_dir)
        if file_or_dir.is_file():
            self.files = [file_or_dir]
        else:
            self.files = [
                str(fpath) for fpath in file_or_dir.glob("*.*")
                if fpath.suffix.lower() in {'.jpg', '.png', '.jpeg'}
            ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        x = self.trans(image_open(fpath))
        return x, 0


class ClasImageset(Dataset):
    def __init__(
        self,
        root: str,
        set_name: str,
        trans: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self.data_root = Path(root)
        if set_name:
            lines = Path(root, set_name).read_text().splitlines()
        else:
            lines = [
                str(p) + '\t0' for p in Path(root).rglob('*.*')
                if p.suffix.lower() in ['.jpg', '.png', '.jpeg']
            ]
        self.dataset = [line_split(line) for line in lines]
        self.trans = trans

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        '''[ch, H, W], target:int'''
        fpath, target = self.dataset[index]
        img = image_open(str(self.data_root / fpath))
        x = self.trans(img)
        return x, target


class LitClasDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers = 0,
        size: Tuple[int, int] = (224, 224),
        predict_file_or_dir: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.predict_file_or_dir = predict_file_or_dir
        self.traindata = None
        self.validdata = None
        self.trans = [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]

    def setup(self, stage: Optional[str] = None):
        traintrans = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            *self.trans,
        ])
        valtrans = transforms.Compose(self.trans)

        if stage in {'fit', 'train'} or stage is None:
            self.traindata = ClasImageset(
                self.data_dir, 'train.txt', trans = traintrans
            )
        if stage in {'validate', 'fit'}:
            self.validdata = ClasImageset(
                self.data_dir, 'valid.txt', trans = valtrans
            )

        if stage in {'predict'}:
            self.predictset = PredictImageset(
                self.predict_file_or_dir,
                valtrans,
            )

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.traindata,
            shuffle = True,
            batch_size = self.batch_size,
            pin_memory = False,
            num_workers = self.num_workers,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validdata,
            shuffle = False,
            batch_size = self.batch_size,
            pin_memory = False,
            num_workers = self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predictset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = False,
            shuffle = False,
        )