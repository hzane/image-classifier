#!/usr/bin/env python
# %%
from pathlib import Path
from typing import Generator, Union

import torch
import hydra
import pytorch_lightning as pl
import omegaconf as oc

from hydra.utils import to_absolute_path as hydra_to_absolute_path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet34
from torchvision.datasets.folder import find_classes
from tqdm import tqdm

from typing import Optional

from dataclasses import dataclass
from omegaconf import MISSING
from trainvit import (
    XartsModule,
)


def iter_images(root: Union[str, Path]) -> Generator[Path, None, None]:
    for pth in Path(root).rglob('*'):
        if pth.suffix in {'.jpg', '.jpeg', '.png'}:
            yield pth
# %%


class ImageFolder(Dataset):
    def __init__(self, root: Union[str, Path], transform = None):
        self.transform = transform
        self.root = hydra_to_absolute_path(root)
        self.image_paths = list(iter_images(root))
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            img = torch.randn(3, 224, 224)
            print(e)
        return img, str(self.image_paths[idx])
# %%


def clean_images(root: str):

    transes = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for img_fn in tqdm(list(iter_images(root))):
        try:
            img = Image.open(img_fn).convert('RGB')
            img = transes(img)
        except Exception as e:
            print(img_fn, e)
            tgt = img_fn.with_suffix(img_fn.suffix + ".bad")
            img_fn.rename(tgt)


# %%
def predict_dir(
    trainer,
    model,
    classes,
    dataloader,
    target: str,
):
    result = trainer.predict(model, dataloader)

    for x in classes:
        Path(target, x).mkdir(0o755, parents = True, exist_ok = True)

    for labels, names in result:
        for label, name in zip(labels, names):
            t = Path(target, classes[label.item()], Path(name).name)
            print(str(t))
            Path(name).rename(t)


@dataclass
class XartsPredictConfig:
    data_dir: str = MISSING
    scheme: str = 'xart3'
    target_dir: Optional[str] = None
    classes_dir: Optional[str] = None
    clean: bool = False
    batch_size: int = 4
    num_workers: int = 4
    gpus: int = 1


@hydra.main(config_path = None, config_name = 'predict')
def predict_cli(conf: XartsPredictConfig) -> None:
    if conf.target_dir is None:
        conf.target_dir = hydra_to_absolute_path(conf.data_dir)

    if conf.clean:
        clean_images(conf.target_dir)

    if conf.classes_dir is None:
        conf.classes_dir = conf.scheme + '.train'

    classes, _ = find_classes(hydra_to_absolute_path(conf.classes_dir))

    model_ckpt = hydra_to_absolute_path(conf.scheme + '.cpkt')
    model = XartsModule.load_from_checkpoint(model_ckpt)
    model.freeze()

    data = ImageFolder(hydra_to_absolute_path(conf.data_dir))
    dataloader = DataLoader(
        data,
        batch_size = conf.batch_size,
        num_workers = conf.num_workers,
        shuffle = False,
    )

    trainer = pl.Trainer(
        gpus = conf.gpus,
    )

    predict_dir(
        trainer,
        model,
        classes,
        dataloader,
        conf.target_dir,
    )


# %%
# python predict.py --scheme=selfshot --root=test --target=. --clean=0
if __name__ == '__main__':
    cs = hydra.core.config_store.ConfigStore.instance()
    cs.store(name = 'predict', node = XartsPredictConfig)

    predict_cli()
