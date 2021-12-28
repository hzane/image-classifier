#!/usr/bin/env -S conda run -n tf python

import onnxruntime
from nudenet.image_utils import load_images
from pathlib import Path
from fire import Fire
from tqdm import tqdm
from typing import Tuple


def classify(model, batch, image_size = (256, 256)):
    try:
        images, _ = load_images(
            batch, image_size = image_size, image_names = batch
        )
        preds = model.run(None, {model.get_inputs()[0].name: images})[0]
        return preds.argmax(1).tolist()
    except Exception as e:
        return [2] * len(batch)


def main(
    file_or_dir: str,
    batch_size = 64,
    size: Tuple[int] = (256, 256),
):
    file_or_dir = Path(file_or_dir)
    to = file_or_dir.with_suffix('.preds.txt')

    if file_or_dir.is_dir():
        file_or_dir = [
            str(path) for path in file_or_dir.rglob('*.*')
            if path.suffix in {'.jpg', '.jpeg', '.png'}
        ]
    else:
        file_or_dir = [str(file_or_dir)]

    batches = [
        file_or_dir[i:i + batch_size]
        for i in range(0, len(file_or_dir), batch_size)
    ]
    ckpt = Path.home() / '.NudeNet/classifier_model.onnx'
    model = onnxruntime.InferenceSession(
        str(ckpt),
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    labels = ['unsafe', 'safe', 'failed']

    with to.open('w') as f:
        for batch in tqdm(batches):
            yhat = classify(model, batch, size)
            preds = [labels[i] for i in yhat]

            for file, label in zip(batch, preds):
                print(file, label, sep = '\t', file = f)


if __name__ == '__main__'    :
    Fire(main)
