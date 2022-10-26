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
from typing import Dict, Any, Optional, Union, Tuple, List


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


def pretrain_weights(pretrain:str, project, root):
    # pretrain.pth/.ckpt
    pretrain, root = Path(pretrain or '.not-exists'), Path(root,project)
    if not pretrain.exists():
        pretrain = root / pretrain
    ckpt = torch.load(pretrain) if pretrain.exists() else None
    if isinstance(ckpt, dict):
        ckpt = ckpt['state_dict']
    return ckpt


def main(
    project: str = 'cat12',
    num_classes: int = 12,
    data_dir: str = 'datasets/hanren',
    train: str = 'train.txt',
    valid: str = 'valid.txt',
    img_size: List[int] = None,
    batch_size: int = 8,
    backbone: str = 'resnet18',
    lr:float = 0.0005,
    max_epochs: int = 5,
    resume: str = None,
    pretrain: str = None,
    root: str = 'outputs',
    use_aug: bool = True,
    sanity: int = 0,
) -> None:
    accu = 64//batch_size if 32>batch_size else 1
    ver = version(backbone, project, root)
    ckpt_path = resume_ckpt(resume, project, root)
    pretrain = pretrain_weights(pretrain, project, root)
    if img_size is None:
        img_size = [224, 224]

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
    # data_dir = data_dir# 'datasets/cats12/data_sets/cat_12'
    train_dl = train_dataloader(
        data_dir,
        train,
        batch_size = batch_size,
        use_aug = use_aug,
        img_size = img_size
    )
    valid_dl = valid_dataloader(
        data_dir, valid, batch_size = batch_size, img_size = img_size
    )

    model = LitClasModule(num_classes, backbone_name = backbone, lr = lr)
    if pretrain:
        model.load_state_dict(pretrain)

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
