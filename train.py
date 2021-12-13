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
from typing import Dict, Any


class LitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # parser.add_argument(
        #     '--experiment', type = str, default = 'baseline'
        # )
        parser.set_defaults({
            'trainer.default_root_dir': 'outputs/baseline',
        })

    def instantiate_trainerx(self, **kwargs: Any) -> Trainer:
        cmd = self.config['subcommand']
        if cmd in {'fit'}:
            exp = self._get(self.config, 'experiment')
            logger = TensorBoardLogger('logs', exp)

            kwargs = {**kwargs, 'logger': logger}
        else:
            kwargs = {**kwargs, 'logger': False}
        return super().instantiate_trainer(**kwargs)

    def save_best(self, ckpt: str) -> None:
        root = Path(self._get(self.config, 'trainer.default_root_dir'))
        if Path(ckpt).exists():
            Path(root, 'best.ckpt').symlink_to(ckpt)


def main():
    checkpoint = ModelCheckpoint(
            save_last = True,
            filename = 'epoch-{epoch}-acc-{valid/acc:.4f}-step-{step}',
            monitor = 'valid/acc',
            mode = 'max', # min
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
