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
import torchmetrics as tm
import timm
import pytorch_lightning as pl
import hydra
import omegaconf as oc

from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from hydra.utils import to_absolute_path as hydra_to_absolute_path

from typing import Any

from pathlib import Path
# %%


@dataclass
class EarlyStoppingConf:
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: str = "early_stop_on"
    min_delta: float = 0.0
    patience: int = 3
    verbose: bool = False
    mode: str = "auto"
    strict: bool = True


@dataclass
class TrainerConf:
    gpus: Any = 1
    check_val_every_n_epoch: int = 1
    min_epochs: int = 1
    log_every_n_steps: int = 10
    resume_from_checkpoint: Any = None
    fast_dev_run: Any = False


@dataclass
class ModelConf:
    vit_name: str = 'vit_tiny_patch16_224'


@dataclass
class DataConf:
    pass


@dataclass
class XArtsConfig:
    trainer: TrainerConf = TrainerConf()
    model: ModelConf = ModelConf()
    data: DataConf = DataConf()
    seed: int = 210820
    scheme: str = 'xart3'
    epochs: int = 5
    num_workers: int = 4
    batch_size: int = 64
    lr: float = 3e-5
    force_save: bool = False
    num_classes: int = 9


class XArtsModule(pl.LightningModule):
    def __init__(self, conf: XArtsConfig):
        super().__init__()
        self.save_hyperparameters(conf)
        self.criterion = nn.CrossEntropyLoss()
        self.model = self.configure_model()
        self.train_accuracy = tm.Accuracy()

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        self.train_accuracy(yhat, y)
        loss = self.criterion(yhat, y)

        self.log('loss/train', loss)
        self.log('accuracy/train', self.train_accuracy, prog_bar = True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr = self.hparams.lr)
        return optimizer

    def configure_model(self, ):
        model = timm.create_model(
            self.hparams.model.vit_name,
            pretrained = True,
            num_classes = self.hparams.num_classes,
        )

        return model


# %%


class XArtsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 16, num_workers = 0):
        super().__init__()
        self.data_dir = hydra_to_absolute_path(data_dir)
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
            pin_memory = False,
            num_workers = self.num_workers,
        )
        return dataloader
# %%


@hydra.main(config_path = None, config_name = 'config')
def xarts_cli(conf: XArtsConfig) -> None:
    pl.seed_everything(conf.seed)
    print(oc.OmegaConf.to_yaml(conf))

    mc = pl.callbacks.ModelCheckpoint(
        monitor='loss/train',
        auto_insert_metric_name = False,
        filename = 'e{epoch}-t{loss/train:.05f}'
    )

    es = pl.callbacks.early_stopping.EarlyStopping(
        monitor='accuracy/train',
        min_delta = 0.0,
        patience = 10,
        mode = 'max',
    )

    en = conf.trainer.fast_dev_run and 1 or conf.trainer.log_every_n_steps
    trainer = pl.Trainer(
        gpus = conf.trainer.gpus,
        max_epochs = conf.num_epochs,
        fast_dev_run = conf.trainer.fast_dev_run,
        log_every_n_steps = en,
        resume_from_checkpoint = conf.trainer.resume_from_checkpoint,
        callbacks = [mc, es],
    )
    module = XArtsModule(conf)
    data = XArtsDataModule(
        conf.scheme + '.train',
        conf.batch_size,
        conf.num_workers,
    )

    trainer.fit(module, data)
    print(mc.best_model_path)


if __name__ == '__main__':
    '''
    python train.py --scheme=jigsaw
    '''
    cs = hydra.core.config_store.ConfigStore.instance()
    cs.store(name = 'config', node = XArtsConfig)

    xarts_cli()
