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
    qa_train_dataloader,
    qa_valid_dataloader,
    LitQualityModule,
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
    max_epochs: int = 3,
    resume: str = None,
    project: str = 'quality',
    root: str = 'outputs',
    sanity: int = 0,
) -> None:
    accu = 64//batch_size if 32>batch_size else 1
    ver = version(backbone, project, root)
    ckpt_path = resume_ckpt(resume, project, root)

    checkpoint = ModelCheckpoint(
        save_last = True,
        filename = 'epoch-{epoch}-loss-{valid/loss:.3f}-{step}',
        monitor = 'valid/loss',
        mode = 'min',  # max
        auto_insert_metric_name = False,
    )
    callbacks = [
        # 全局 累计patience，所以最好是多个 lr-scheduler 的 patience
        EarlyStopping(patience = 20, monitor = 'valid/loss'),
        LearningRateMonitor(logging_interval = 'epoch'),
        checkpoint,
    ]
    data_dir = 'datasets/hanren'
    train_dl = qa_train_dataloader(data_dir, 'aip-quality-train-mini.tsv', batch_size = batch_size)
    valid_dl = qa_valid_dataloader(data_dir, 'aip-quality-valid-mini.tsv', batch_size = batch_size)

    model = LitQualityModule(2, backbone_name = backbone)
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
