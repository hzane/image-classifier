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


<<<<<<< Updated upstream
class LitFacePredictCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            '--ckpt_path', type = str, default = 'cat12-convit'
        )
        parser.add_argument('--save-to', type = str, default = None)

    def pretrained_path(self) -> Optional[str]:
        ckpt_path = self._get(self.config, 'ckpt_path')
        if not ckpt_path:
            return None

        ckpt = Path(ckpt_path)
        if ckpt.is_symlink():
            ckpt = ckpt.readlink()

        if ckpt.exists() and ckpt.is_file():
            return ckpt_path

        ckpt = Path('outputs', ckpt_path, 'best.ckpt')
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
=======
def iter_images(root: Union[str, Path]) -> Generator[Path, None, None]:
    for pth in Path(root).rglob('*'):
        if pth.suffix in {'.jpg', '.jpeg', '.png'}:
            yield pth
# %%


class ImageFolder(Dataset):
    def __init__(self, root: Union[str, Path], transform = None):
        super().__init__()
        self.transform = transform
        self.root = hydra_to_absolute_path(root)
        self.image_paths = list(iter_images(root))
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            img = torch.randn(3, 224, 224)
            print(e)
        return img, str(self.image_paths[idx])
# %%


def clean_images(root: str):

    transes = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for img_fn in tqdm(list(iter_images(root))):
        try:
            img = Image.open(img_fn).convert('RGB')
            img = transes(img)
        except Exception as e:
            print(img_fn, e)
            tgt = img_fn.with_suffix(img_fn.suffix + ".bad")
            img_fn.rename(tgt)


# %%
def predict_dir(
    trainer,
    model,
    classes,
    dataloader,
    target: str,
):
    result = trainer.predict(model, dataloader)

    for x in classes:
        Path(target, x).mkdir(0o755, parents = True, exist_ok = True)

    for labels, names in result:
        for label, name in zip(labels, names):
            t = Path(target, classes[label.item()], Path(name).name)
            print(str(t))
            Path(name).rename(t)


@dataclass
class XartsPredictConfig:
    data_dir: str = MISSING
    scheme: str = 'xart3'
    target_dir: Optional[str] = None
    classes_dir: Optional[str] = None
    clean: bool = False
    batch_size: int = 120
    num_workers: int = 2
    gpus: int = 1


@hydra.main(config_path = None, config_name = 'predict')
def predict_cli(conf: XartsPredictConfig) -> None:
    if conf.target_dir is None:
        conf.target_dir = hydra_to_absolute_path(conf.data_dir)

    if conf.clean:
        clean_images(conf.target_dir)

    if conf.classes_dir is None:
        conf.classes_dir = conf.scheme + '.train'

    classes_dir = Path(__file__).with_name(conf.scheme).with_suffix('.train')
    classes, _ = find_classes(hydra_to_absolute_path(classes_dir))
    model_ckpt = Path(__file__).with_name(conf.scheme).with_suffix('.ckpt')

    model = XartsModule.load_from_checkpoint(model_ckpt)
    model.freeze()

    data = ImageFolder(hydra_to_absolute_path(conf.data_dir))
    dataloader = DataLoader(
        data,
        batch_size = conf.batch_size,
        num_workers = conf.num_workers,
        shuffle = False,
    )

    trainer = pl.Trainer(
        gpus = conf.gpus,
        logger = False,
    )
>>>>>>> Stashed changes

    predictions = trainer.predict(
        cli.model, cli.datamodule, ckpt_path = cli.pretrained_path()
    )

    dataset = cli.datamodule.predictset
    files = dataset.files
    predictions = torch.cat(predictions).flatten()
    for fname, label in zip(files, predictions):
        print(fname, label.item(), sep = '\t')
    print(len(predictions), len(dataset))

<<<<<<< Updated upstream
=======
# %%
# python predict.py scheme=selfie data_dir=test
if __name__ == '__main__':
    cs = hydra.core.config_store.ConfigStore.instance()
    cs.store(name = 'predict', node = XartsPredictConfig)
>>>>>>> Stashed changes

# python predict.py --config=configs/cat12-convit.yaml \
#   --ckpt_path=cat12-convit \
#   --data.predict_file_or_dir=datasets/cats12/data_sets/cat_12/cat_12_test
if __name__ == '__main__':
    main()
