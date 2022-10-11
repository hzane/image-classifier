#!/usr/bin/env python

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from xclas import (
    train_dataloader,
    valid_dataloader,
    LitClasModule,
)
from pathlib import Path
from typing import Dict, Any, Optional



def main():
    checkpoint = ModelCheckpoint(
        save_last = True,
        filename = 'epoch-{epoch}-acc-{valid/acc:.4f}-step-{step}',
        monitor = 'valid/acc',
        mode = 'max',  # min
        auto_insert_metric_name = False,
    )
    callbacks = [
        # 全局 累计patience，所以最好是多个 lr-scheduler 的 patience
        EarlyStopping(patience = 20, monitor = 'valid/loss'),
        LearningRateMonitor(logging_interval = 'epoch'),
        checkpoint,
    ]
    data_dir = 'datasets/cats12/data_sets/cat_12'
    train_dl = train_dataloader(data_dir, 'train.txt', batch_size = 4)
    valid_dl = valid_dataloader(data_dir, 'valid.txt', batch_size = 4)

    model = LitClasModule(12)
    trainer = Trainer(
        # limit_train_batches = 100,
        logger = WandbLogger(name='cat12-resnet18'),
        max_epochs = 100,
        callbacks = callbacks,
        accelerator = 'gpu',
        devices = 1,
    )
    trainer.fit(model, train_dl, valid_dl)



if __name__ == "__main__":
    '''
    python train.py --help
    python train.py fit --config=configs/cat12-convit.yaml
    '''
    main()
