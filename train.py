#!/usr/bin/env python

import torch
import _env as env
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint
)
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from xclas import LitClasDataModule, LitClasModule
from pathlib import Path
from typing import Dict, Any, Optional


class LitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.set_defaults({
            'trainer.default_root_dir': 'outputs/baseline',
        })

    def save_best(self, ckpt: str) -> None:
        trainer = self._get(self.config, 'trainer', {})
        root = trainer.get('default_root_dir', 'outputs')
        if not root:
            return
        root, ckpt = Path(root), Path(ckpt)

        if Path(ckpt).exists() and root.exists():
            Path(root, 'best.ckpt').symlink_to(ckpt.relative_to(root))


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

    cli = LitCLI(
        LitClasModule,
        LitClasDataModule,
        trainer_defaults = dict(
            gpus = 1,
            callbacks = callbacks,
        ),
        parser_kwargs = dict(
            fit = dict(default_config_files = ["configs/default.yaml"])
        ),
    )
    cli.save_best(checkpoint.best_model_path)


if __name__ == "__main__":
    '''
    python train.py --help
    python train.py fit --config=configs/cat12-convit.yaml
    '''
    main()
