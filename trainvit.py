#!/usr/bin/env python
# %%
import math
from copy import deepcopy
from time import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from vit_pytorch.efficient import ViT
from linformer import Linformer

from fire import Fire
from pathlib import Path


def train_model(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    checkpoint,
):
    assert checkpoint is not None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_data = len(dataloader.dataset)
    print(f'best-loss: {checkpoint.loss:.4f}, best-acc: {checkpoint.acc:.4f}')

    for epoch in range(epochs):
        t0 = time()

        model.train()
        running_loss, running_acc = 0, 0.
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_acc += torch.sum(y == preds).item()

        running_loss = running_loss / n_data
        running_acc = running_acc / n_data

        if running_acc > checkpoint.acc or (running_acc == checkpoint.acc and
                                            running_loss < checkpoint.loss):
            checkpoint.acc = running_acc
            checkpoint.loss = running_loss
            checkpoint.model_weights = deepcopy(model.state_dict())

        print(
            f'EPOCH: {epoch}, {time()-t0:.0f} secs, Loss: {running_loss:.4f}, '
            f'Acc: {running_acc:.4f}'
        )
        scheduler.step()

    checkpoint.optimizer_state = optimizer.state_dict()
    checkpoint.scheduler_state = scheduler.state_dict()
    # model.load_state_dict(best_model_weights)
    return model


# %%


def train(
    scheme: str,
    epochs: int = 10,
    num_workers: int = 2,
    batch_size: int = 128,
    lr = 3e-5,
    force_save: bool = False
):
    assert scheme is not None
    root_dir = scheme + '.train'
    checkpoint_path = scheme + '.vit.tar'

    transes = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    data = ImageFolder(root_dir, transform = transes)

    dataloader = DataLoader(
        data,
        shuffle = True,
        batch_size = batch_size,
        pin_memory = True,
        num_workers = num_workers,
    )

    print(', '.join(data.classes))

    # %%
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
        num_classes = len(data.classes),
        transformer = efficient_transformer,
        channels = 3,
    ).cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr = lr)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size = 5,
        gamma = 0.7,
    )

    checkpoint = Checkpoint(
        model.state_dict(),
        optimizer.state_dict(),
        exp_lr_scheduler.state_dict(),
        data.classes,
    )
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint.model_weights)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        exp_lr_scheduler.load_state_dict(checkpoint.scheduler_state)

    if checkpoint is not None and force_save:
        checkpoint.acc = 0.

    train_model(
        model,
        dataloader,
        criterion,
        optimizer,
        exp_lr_scheduler,
        epochs,
        checkpoint,
    )
    torch.save(checkpoint, checkpoint_path)


@dataclass
class Checkpoint:
    model_weights : any
    optimizer_state : any
    scheduler_state : any
    classes: list
    loss: float = float('inf')
    acc : float = 0.


if __name__ == '__main__':
    '''
    python train.py --scheme=jigsaw
    '''
    Fire(train)
