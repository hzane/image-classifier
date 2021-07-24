#!/usr/bin/env python
# %%
from pathlib import Path
from typing import Generator, Union

import torch
import torch.nn as nn
from fire import Fire
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet34
from tqdm import tqdm
from trainvit import Checkpoint
from vit_pytorch.efficient import ViT
from linformer import Linformer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def iter_images(root: Union[str, Path]) -> Generator[Path, None, None]:
    for p in Path(root).rglob("*.jpg"):
        yield p
    for p in Path(root).rglob("*.png"):
        yield p


# %%


class ImageFolder(Dataset):
    def __init__(self, root: Union[str, Path], transform = None):
        self.transform = transform
        self.root = root
        self.image_paths = list(iter_images(root))
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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


def load_model(checkpoint):
    if not checkpoint.endswith('.tar'):
        checkpoint += '.vit.tar'

    if not Path(checkpoint).exists():
        checkpoint = Path(__file__).resolve().parent / checkpoint

    cp = torch.load(checkpoint)

    efficient_transformer = Linformer(
        dim = 256,
        seq_len = 49 + 1,
        depth = 12,
        heads = 8,
        k = 64,
    )

    model = ViT(
        dim = 256,
        image_size = 224,
        patch_size = 32,
        num_classes = len(cp.classes),
        transformer = efficient_transformer,
        channels = 3,
    ).cuda()

    model.load_state_dict(cp.model_weights)
    classes = cp.classes

    return model, classes


def predict(model, dataloader):
    model.eval()

    files, preds = [], []
    for x, fn in tqdm(dataloader):
        x = x.to(device)
        output = model(x)  # (batch_size, 2)
        pred = torch.argmax(output, 1)
        files += [f for f in fn]
        preds += [p.item() for p in pred]

    return list(zip(files, preds))


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
        model,
        classes,
        root: str,
        target: str,
        num_workers ,
        batch_size ,
        clean: bool = False,
):
    if clean:
        clean_images(root)

    if target is None:
        target = root

    data = ImageFolder(root)
    print(f'{root} has {len(data)} images')
    if len(data) == 0:
        return

    dataloader = DataLoader(
        data,
        shuffle = False,
        batch_size = batch_size,
        num_workers = num_workers,
    )
    with torch.no_grad():
        result = predict(model, dataloader)

    for x in classes:
        Path(target, x).mkdir(0o755, parents=True, exist_ok = True)

    for fn, label in result:
        print(fn, classes[label])
        t = Path(target, classes[label], Path(fn).name)
        Path(fn).rename(t)


def predict_dirs(
        root: str,
        clean: bool = False,
        target: str = None,
        scheme: str = None,
        num_workers: int = 2,
        batch_size: int = 100,
):
    assert scheme is not None
    model, classes = load_model(scheme)

    predict_dir(model, classes, root, target, num_workers, batch_size, clean)


# %%
# python predict.py --scheme=selfshot --root=test --target=. --clean=0
if __name__ == '__main__':
    Fire(predict_dirs)
