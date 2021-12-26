#!/usr/bin/env -S conda run -n tf python

import numpy as np
import cv2
import onnxruntime
import json
import requests
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from fire import Fire
from typing import List, Union


def preprocess_image(
    image_path,
    min_side = 800,
    max_side = 1333,
):
    image = read_image_bgr(image_path)
    image = _preprocess_image(image)
    image, scale = resize_image(
        image, min_side = min_side, max_side = max_side
    )
    return image, scale


def read_image_bgr(path):
    """ Read an image in BGR format.
    Args
        path: Path to the image.
    """
    if isinstance(path, (str, Path)):
        image = np.array(Image.open(path).convert("RGB"))
    else:
        path = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
        image = np.array(Image.fromarray(path))

    return image[:, :, ::-1]


def _preprocess_image(x, mode = "caffe"):
    x = x.astype(np.float32)

    if mode == "tf":
        x /= 127.5
        x -= 1.0
    elif mode == "caffe":
        x -= np.array([103.939, 116.779, 123.68])

    return x


def compute_resize_scale(image_shape, min_side = 800, max_side = 1333):
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    scale = min_side / smallest_side

    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side = 800, max_side = 1333):
    scale = compute_resize_scale(
        img.shape, min_side = min_side, max_side = max_side
    )

    img = cv2.resize(img, None, fx = scale, fy = scale)

    return img, scale


def detect(model, classes, image_path, fast: bool = False) -> List[dict]:
    image, scale = preprocess_image(image_path)
    args_o = [s.name for s in model.get_outputs()]
    args_i = {model.get_inputs()[0].name: image[None, :, :, :]}

    boxes, labels, scores = model.run(args_o, args_i)
    boxes, labels, scores = boxes[0], labels[0], scores[0]

    min_prob = 0.6
    boxes /= (image.shape[:2] * 2)

    results = [
        dict(
            box = box.tolist(),
            score = score.item(),
            label = classes[label.item()],
            path = str(image_path),
        ) for box, score, label in zip(boxes, scores, labels)
        if score > min_prob
    ]

    return results


def download_file(url: str, to: Union[str, Path])->None:
    to = Path(to)
    if to.exists():
        print('using cached file', to)
        return

    tmp = to.with_suffix('.tmp')
    resp = requests.get(url, stream = True)

    block_size = 1024*1024
    total_length = int(resp.headers.get('content-length', 0))
    progress_bar = tqdm(total = total_length, unit = 'iB', unit_scale = True)

    with tmp.open('wb') as file:
        for data in resp.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if progress_bar.n == total_length:
        tmp.rename(to)


def prepare_weight(model_name: str)->None:
    root = Path.home() / '.NudeNet'
    url = 'https://github.com/notAI-tech/NudeNet/releases/download/v0'

    w = root / f'{model_name}_checkpoint.onnx'
    wurl = f'{url}/{w.name}'
    download_file(wurl, w)

    c = root / f'{model_name}_classes'
    curl = f'{url}/{c.name}'
    download_file(curl, c)


def prepare_weights()->None:
    # prepare_weight('detector_v2_base')
    prepare_weight('detector_v2_default')


def main(file_or_dir: str = None, out: str = None):
    prepare_weights()

    if file_or_dir is None:
        file_or_dir = 'whats.train/psed.a/3C49E51BF3B55D51B9582CE4735DDE3CDA0523C7-t.jpg'
    file_or_dir = Path(file_or_dir)

    if file_or_dir.is_file():
        images = [file_or_dir]
    else:
        images = [
            image for image in file_or_dir.rglob('*.*')
            if image.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        ]

    if not out:
        out = file_or_dir.name

    out = Path(out).with_suffix('.json')
    out.parent.mkdir(parents = True, exist_ok = True)

    ckpt = Path.home() / '.NudeNet/detector_v2_default_checkpoint.onnx'
    model = onnxruntime.InferenceSession(str(ckpt))
    classes = Path.home().joinpath('.NudeNet/classes'
                                  ).read_text().splitlines()

    with out.open('w', encoding = 'utf-8') as out:
        for image_path in tqdm(images):
            for r in detect(model, classes, image_path):
                json.dump(r, out, ensure_ascii = False)
                out.write('\n')


if __name__ == '__main__':
    # run as standalone script:
    # nu-detect.py --file_or_dir=... --out=...
    Fire(main)
