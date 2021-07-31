#!/usr/bin/env python
# %%
from dataclasses import dataclass, asdict

# %%
import math
from copy import deepcopy
from time import time

from typing import Optional

import torch
import torch.nn as nn

from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from vit_pytorch.efficient import ViT
from linformer import Linformer

# from fire import Fire
from pathlib import Path
# %%


class XArtsModule(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        self.model = self.configure_model()

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        # preds = yhat.argmax(1)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr = self.hparams.lr)
        return optimizer

    def configure_model(self,):
        efficient_transformer = Linformer(
            dim = 256,
            seq_len = 49 + 1,
            depth = 12,
            heads = 8,
            k = 64,
        )

        model = ViT(
            dim = 256,
            image_size = 224,
            patch_size = 32,
            num_classes = self.hparams.num_classes,
            transformer = efficient_transformer,
            channels = 3,
        )
        return model


# %%


class XArtsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 16, num_workers = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = None

    def setup(self, stage: Optional[str] = None):
        transes = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data = ImageFolder(self.data_dir, transform = transes)

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.data,
            shuffle = True,
            batch_size = self.batch_size,
            pin_memory = True,
            num_workers = self.num_workers,
        )
        return dataloader

    def val_dataloader_no(self):
        pass

    def test_dataloader_no(self):
        pass
# %%


@dataclass
class Experiment:
    scheme: str
    epochs: int = 10
    num_workers: int = 4
    batch_size: int = 128
    lr: float = 3e-5
    force_save: bool = False
    num_classes: int = 9

    def as_dict(self):
        return asdict(self)

    def train(self):
        module = XArtsModule(**self.as_dict())
        trainer = pl.Trainer(gpus = 1)
        xarts = XArtsDataModule(self.scheme + '.train', self.batch_size,
                                self.num_workers, )

        trainer.fit(module, xarts)

# %%


if __name__ == '__main__':
    '''
    python train.py --scheme=jigsaw
    '''
    from fire import Fire

    Fire(Experiment)
