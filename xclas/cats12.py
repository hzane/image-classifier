#!/usr/bin/env python

from random import shuffle
from pathlib import Path


def main(
    root: str = 'datasets/cats12/data_sets/cat_12',
    train_ratio: float = 0.8
):
    root = Path(root)
    meta = Path(root, 'train_list.txt')
    lines = meta.read_text().splitlines()
    shuffle(lines)
    train_num = int(len(lines) * train_ratio)
    trainset = lines[:train_num]
    validset = lines[train_num:]
    Path(root, 'train.txt').write_text('\n'.join(trainset))
    Path(root, 'valid.txt').write_text('\n'.join(validset))    


if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)