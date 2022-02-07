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
from typing import List, Union, Tuple


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
        try:
            image = np.array(Image.open(path).convert("RGB"))
        except Exception:
            image = np.zeros((100, 100, 3), dtype = np.uint8)
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


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # heightx
    return y


def detect(model, classes, image_path, fast: bool = False) -> List[dict]:
    image, scale = preprocess_image(image_path)
    args_o = [s.name for s in model.get_outputs()]
    args_i = {model.get_inputs()[0].name: image[None, :, :, :]}

    boxes, labels, scores = model.run(args_o, args_i)
    boxes, labels, scores = boxes[0], labels[0], scores[0]

    min_prob = 0.6
    boxes /= (image.shape[:2] * 2)

    # results = [
    #     dict(
    #         box = xyxy2xywh(box).tolist(),
    #         score = score.item(),
    #         label = classes[label.item()],
    #         labelid = label.item(),
    #         path = str(image_path),
    #         imgsz = image.shape[:2],
    #     ) for box, score, label in zip(boxes, scores, labels)
    #     if score > min_prob
    # ]
    # _ = results
    lines = [
        f'{str(label.item())}  {" ".join([str(f) for f in xyxy2xywh(box)])}'
        for box, score, label in zip(boxes, scores, labels)
        if score > min_prob
    ]

    return lines


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


def prepare_weight(model_name: str) -> Tuple[str, List[str]]:
    root = Path.home() / '.NudeNet'
    url = 'https://github.com/notAI-tech/NudeNet/releases/download/v0'

    w = root / f'{model_name}_checkpoint.onnx'
    wurl = f'{url}/{w.name}'
    download_file(wurl, w)

    c = root / f'{model_name}_classes'
    curl = f'{url}/{c.name}'
    download_file(curl, c)
    return str(w), c.read_text().splitlines()


def prepare_weights() -> Tuple[str, List[str]]:
    # prepare_weight('detector_v2_base')
    return prepare_weight('detector_v2_default')


def main(file_or_dir: str = None, out: str = None):
    ckpt, classes = prepare_weights()
    model = onnxruntime.InferenceSession(
        ckpt,
        providers = [
            'CUDAExecutionProvider',
            # 'TensorrtExecutionProvider',
            'CPUExecutionProvider',
        ]
    )

    file_or_dir = Path(file_or_dir)

    if file_or_dir.is_file():
        images = [file_or_dir]
    else:
        images = [
            image for image in file_or_dir.rglob('*.*')
            if image.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        ]

    if not out:
        out = file_or_dir.parent

    for image_path in tqdm(images):
        rpath = image_path.relative_to(out)
        out_path = out / 'labels' / rpath.with_suffix('.txt')
        if out_path.exists():
            continue
        out_path.parent.mkdir(parents = True, exist_ok = True)

        results = detect(model, classes, image_path)
        out_path.write_text('\n'.join(results))

    # {"box": xyxy
    # [1.2085703611373901, 0.5202627778053284, 1.4289566278457642, 0.6306127309799194],
    # "score": 0.6090637445449829,
    # "label": "EXPOSED_FEET",
    # "path": "xarts6.train/covers.a/d7764bd43d3b8ed3715c60d064dd30c47780c7b9-t.jpg"}


if __name__ == '__main__':
    # run as standalone script:
    # nu-detect.py --file_or_dir=... --out=...
    Fire(main)
