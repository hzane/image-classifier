import torch
import torch.nn as nn
import timm
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
from typing import Any,List
from .datamodules import ImageDataset, FileDataset



def qa_train_dataloader(
    data_dir,
    fileset,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: List[int] = None,    
) -> DataLoader:
    fileset = FileDataset(fileset, data_dir, target_is_int = False)
    dataset = ImageDataset(fileset, img_size, use_aug = False)
    ret = DataLoader(
        dataset,
        batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True
    )
    return ret


def qa_valid_dataloader(
    data_dir,
    fileset,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: List[int] = None
) -> DataLoader:
    fileset = FileDataset(fileset, data_dir, target_is_int = False)
    dataset = ImageDataset(fileset, img_size)
    ret = DataLoader(
        dataset,
        batch_size,
        num_workers = num_workers,
        shuffle = False,
        pin_memory = False
    )
    return ret


def qa_predict_dataloader(
    root,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: List[int] = None,
) -> DataLoader:
    fileset = FileDataset(root, root = None, target_is_int = False)
    dataset = ImageDataset(fileset, img_size)
    return DataLoader(
        dataset,
        batch_size,
        num_workers = num_workers,
        pin_memory = False,
        shuffle = False
    )


class LitQualityModule(LightningModule):
    def __init__(
        self,
        num_indexes: int = 2,
        backbone_name: str = 'resnet18',
        lr: float = 1.e-3,
        scheduler_patience: int = 5,
        lr_reduce_factor: float = 0.33,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = self.configure_criterion()
        self.model = self.configure_model(backbone_name, num_indexes)

    def configure_optimizers(self):
        hp = self.hparams
        optimizer = torch.optim.Adam(self.parameters(), lr = hp.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience = hp.scheduler_patience,
            factor = hp.lr_reduce_factor,
        )

        return dict(
            optimizer = optimizer,
            lr_scheduler = dict(
                scheduler = scheduler,
                monitor = 'valid/loss',
            ),
        )

    def configure_model(self, backbone: str, num_classes: int):
        model = timm.create_model(
            backbone,
            pretrained = True,
            num_classes = num_classes,
        )

        return model

    def configure_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def predict_step(
        self, batch, batch_idx: int, dataloader_idx: int = None
    ):
        x, names = batch
        yhat = self(x)
        return yhat.argmax(1)

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log(
            'train/loss', loss.item(), on_step = False, on_epoch = True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log('test/loss', loss)

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        yhat = self(x)

        loss = self.criterion(yhat, y)
        self.log(
            'valid/loss',
            loss,
            on_step = False,
            on_epoch = True,
            prog_bar = True
        )

        return loss
