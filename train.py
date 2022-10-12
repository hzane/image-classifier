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
from typing import Dict, Any, Optional, Union, Tuple


def resume_ckpt(resume:str, project:str, root: str)->Tuple[str, str]:
    root = Path(root, project)
    ckpt = root.joinpath(resume or '', 'checkpoints/last.ckpt')
    ckpt = str(ckpt) if ckpt.exists() else None
    return ckpt #, resume if ckpt else None

def version(backbone:str, project:str, root: str):
    tpath, idx = Path(root, project, f'{backbone}-1v'), 1
    while tpath.exists():
        idx = idx+2
        tpath = tpath.with_name(f'{backbone}-{idx}v')
    return tpath.name


def main(
    batch_size: int = 8,
    backbone: str = 'resnet18',
    max_epochs: int = 5,
    resume: str = None,
    project: str = 'cat12',
    root: str = 'outputs',
    use_aug: bool = True,
    sanity: int = 0,
) -> None:
    accu = 64//batch_size if 32>batch_size else 1
    ver = version(backbone, project, root)
    ckpt_path = resume_ckpt(resume, project, root)

    checkpoint = ModelCheckpoint(
        save_last = True,
        filename = 'epoch-{epoch}-acc-{valid/acc:.3f}-{step}',
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
    train_dl = train_dataloader(data_dir, 'train.txt', batch_size = batch_size, use_aug = use_aug)
    valid_dl = valid_dataloader(data_dir, 'valid.txt', batch_size = batch_size)

    model = LitClasModule(12, backbone_name = backbone)
    trainer = Trainer(        
        num_sanity_val_steps = sanity,
        default_root_dir=root,
        accumulate_grad_batches=accu,
        logger = WandbLogger(version=ver, project=project),
        max_epochs = max_epochs,
        callbacks = callbacks,
        accelerator = 'gpu',
        devices = 1,
    )
    trainer.fit(model, train_dl, valid_dl, ckpt_path = ckpt_path)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
