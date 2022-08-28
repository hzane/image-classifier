#!/usr/bin/env python

import pandas as pd
from pathlib import Path
from xxhash import xxh3_128_hexdigest
from fire import Fire
from tqdm import tqdm


def hashmbnx(fpath: str, block_size: int = 2**20):
    dig = None
    try:
        dig = hashmb(fpath, block_size)
    except Exception as e:
        print(fpath, e)
        dig = None
    return dig


def hashmb(fpath: str, block_size):
    dig = None
    with open(fpath, "rb") as f:
        chunk = f.read(block_size)
        dig = xxh3_128_hexdigest(chunk)

    return dig


def rename(to: Path):
    stem = to.stem
    idx = 1
    while to.exists():
        to = to.with_name(f'{stem}-{idx}{to.suffix}')
        idx = idx + 1
    return to


def movefile(fpath, digest,root):
    fpath = Path(fpath)
    to = root / f'{digest[:2]}' /fpath.name
    to = rename(to)
    to.parent.mkdir(parents=True, exist_ok = True)

    fpath.rename(to)


def main(root: str, block_size: int = 2**16, drop:bool = False):
    root = Path(root)
    exts = {'.json', '.txt', '.cue'}

    if block_size<16: # MB
        block_size = block_size * 2**20
    elif block_size< 1024:  # KB
        block_size = block_size * 2**10

    files = [fpath for fpath in root.glob("**/*") if fpath.is_file() and fpath.suffix not in exts]

    hashes, sizes = [], []
    for fpath in tqdm(files):
        sz = fpath.stat().st_size
        sizes.append(sz)
        dig = hashmbnx(fpath, block_size)
        hashes.append(dig)

    files = [str(f) for f in files]
    dataset = pd.DataFrame.from_dict(dict(file=files, digest=hashes, bytes=sizes))

    to = root.with_suffix('.dupfind.csv')
    dataset.to_csv(to, index=False, header=False)

    keep = 'first' if drop else False
    dups = dataset.duplicated(subset=['digest', 'bytes'], keep=keep)
    dups = dataset[dups]

    for row in tqdm(dups.itertuples(), total=len(dups)):
        movefile(row.file, row.digest, root.with_name(f'{root.name}-dups'))


if __name__ == '__main__':
    Fire(main)
