#!/usr/bin/env python

from random import shuffle
from pathlib import Path
from fire import Fire


def meta_from_path(path: Path, root: Path):
    path = path.relative_to(root)
    label = path.parts[0]
    return str(path), label


def main(
    root: str = 'datasets/cats12/data_sets/cat_12',
    train_ratio: float = 0.8
):
    root = Path(root)
    meta = [ meta_from_path(p, root) for p in Path(root).rglob('*.*') if p.suffix.lower() in ['.jpg', '.png', '.jpeg'] ]

    labels = sorted(set(label for _, label in meta))
    Path(root, 'labels.tsv').write_text('\n'.join([f'{i}\t{l}' for i, l in enumerate(labels)]))

    labeldict = {l: i for i, l in enumerate(labels)}
    meta = [f'{fpath}\t{labeldict[label]}' for fpath, label in meta]
    shuffle(meta)

    train_num = int(len(meta) * train_ratio)
    trainset = meta[:train_num]
    Path(root, 'train.txt').write_text('\n'.join(trainset))

    validset = meta[train_num:]
    Path(root, 'valid.txt').write_text('\n'.join(validset))
    print(f'{len(trainset)} train images, {len(validset)} valid images')


if __name__ == '__main__':
    Fire(main)
