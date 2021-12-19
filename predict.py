#!/usr/bin/env python

import torch
import _env as env
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning import Trainer
from xclas import LitClasModule, LitClasDataModule
from pathlib import Path
from typing import Any, Optional
import warnings
warnings.filterwarnings(
    action = "ignore",
    message =
    ".*Lightning couldn't infer the indices fetched for your dataloader.*",
)


class LitFacePredictCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            '--ckpt_path', type = str, default = 'cat12-convit'
        )
        # parser.add_argument('--save_to', type = str, default = None)
        parser.add_argument(
            '--no-resplit', dest = 'resplit', action = 'store_false'
        )

    def resplit(self)->bool:
        return self._get(self.config, 'resplit', True)

    def pretrained_path(self) -> Optional[str]:
        ckpt_path = self._get(self.config, 'ckpt_path')
        if ckpt_path:
            return ckpt_path

        trainer = self._get(self.config, 'trainer', {})
        root = trainer.get('default_root_dir', 'outputs/baseline')

        ckpt = Path(root, 'best.ckpt')
        if ckpt.exists():
            return ckpt

        ckpts = [
            p for p in Path('outputs', ckpt_path).rglob('*.ckpt')
            if p.stem.startswith('epoch')
        ]
        if not len(ckpts):
            ckpts = [
                p for p in Path(ckpt_path).rglob('*.ckpt')
                if p.stem.startswith('epoch')
            ]
        if not len(ckpts):
            return None
        ckpts = sorted(
            ckpts,
            key = lambda p: float(p.stem.split('-')[3]),
        )
        return str(ckpts[-1])


def main():
    cli = LitFacePredictCLI(
        LitClasModule,
        LitClasDataModule,
        run = False,
        save_config_callback = None,
        trainer_defaults = dict(
            gpus = 1,
            logger = False,
        ),
    )

    trainer = cli.trainer

    predictions = trainer.predict(
        cli.model, cli.datamodule, ckpt_path = cli.pretrained_path()
    )

    datamodule = cli.datamodule
    labels = datamodule.read_labels()

    dataset = datamodule.predictset
    files = dataset.files
    predictions = torch.cat(predictions).flatten()

    saveto = None
    if cli.resplit():
        saveto = Path(datamodule.predict_file_or_dir)
        for label in labels:
            Path(saveto, label).mkdir(exist_ok = True, parents = True)

    for fname, label in zip(files, predictions):
        label = labels[label.item()]
        print(fname, label, sep = '\t')
        if saveto:
            fname = Path(fname)

            dest = saveto / label
            fname.rename(dest / fname.name)

    print(len(predictions), len(dataset))


# python predict.py --config=configs/cat12-convit.yaml \
#   --ckpt_path=cat12-convit \
#   --data.predict_file_or_dir=datasets/cats12/data_sets/cat_12/cat_12_test \
#   --resplit
if __name__ == '__main__':
    main()
