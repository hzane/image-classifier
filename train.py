#!/usr/bin/env python
# %%
import math
from copy import deepcopy
from time import time

import torch
import torch.nn as nn

from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34

from fire import Fire


def train_model(
        model,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        epochs,
        checkpoint,
        save,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_weights = deepcopy(model.state_dict())
    best_loss = float('inf')
    best_acc = 0.
    n_data = len(dataloader.dataset)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_loss = checkpoint['loss']
        best_acc = checkpoint['acc']
        print(f'best-loss: {best_loss:.4f}, best-acc: {best_acc:.4f}')

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

        if running_acc > best_acc or (running_acc == best_acc and
                                      running_loss < best_loss):
            best_loss = running_loss
            best_acc = running_acc
            best_model_weights = deepcopy(model.state_dict())
            save(running_loss, running_acc)

        print(
            f'EPOCH: {epoch}, {time()-t0:.0f} secs, Loss: {running_loss:.4f}, '
            f'Acc: {running_acc:.4f}'
        )
        scheduler.step()

    model.load_state_dict(best_model_weights)
    return model, best_loss, best_acc


# %%


def train(scheme: str, epochs: int = 10, num_workers: int = 2, batch_size: int = 100, lr = 0.01, force_save: bool = False):
    assert scheme is not None

    root_dir = scheme + '.train'
    checkpoint_path = scheme + '.resnet34.tar'

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
        pin_memory=True,
        num_workers = num_workers,
    )

    print(', '.join(data.classes))

    # %%
    model = resnet34(pretrained = False, num_classes=len(data.classes)).cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr = lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.5)

    try:
        checkpoint = torch.load(checkpoint_path)
    except Exception:
        checkpoint = None

    if checkpoint is not None and force_save:
        checkpoint['acc'] = 0.

    def save(loss, acc):
        torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': exp_lr_scheduler.state_dict(),
        'loss': loss,
        'acc': acc,
        'classes': data.classes,
        }, checkpoint_path)

    modek, loss, acc = train_model(
        model,
        dataloader,
        criterion,
        optimizer,
        exp_lr_scheduler,
        epochs,
        checkpoint,
        save,
    )

    # save(loss, acc)



if __name__ == '__main__':
    '''
    python train.py --scheme=jigsaw
    '''
    Fire(train)
