import torch
import torch.nn as nn
import timm
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

from typing import Any


class LitClasModule(LightningModule):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = 'resnet18',
        lr: float = 1.e-3,
        scheduler_patience: int = 5,
        lr_reduce_factor: float = 0.33,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = self.configure_criterion()
        self.model = self.configure_model(backbone_name, num_classes)
        self.accuracy = Accuracy(num_classes = num_classes)

    def configure_optimizers(self):
        hp = self.hparams
        optimizer = torch.optim.Adam(self.parameters(), lr = hp.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience = hp.scheduler_patience,
            factor = hp.lr_reduce_factor,
        )

        return dict(
            optimizer = optimizer,
            lr_scheduler = dict(
                scheduler = scheduler,
                monitor = 'valid/loss',
            ),
        )

    def configure_model(self, backbone: str, num_classes: int):
        model = timm.create_model(
            backbone,
            pretrained = True,
            num_classes = num_classes,
        )

        return model

    def configure_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def predict_step(
        self, batch, batch_idx: int, dataloader_idx: int = None
    ):
        x, names = batch
        yhat = self(x)
        return yhat.argmax(1)

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log(
            'train/loss', loss.item(), on_step = False, on_epoch = True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log('test/loss', loss)

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        yhat = self(x)
        self.accuracy.update(yhat ,y)
        self.prec.update(yhat, y)
        self.recall.update(yhat, y)

        loss = self.criterion(yhat, y)
        self.log_dict(
            {
                'valid/acc': self.accuracy,
                'valid/loss': loss
            },
            on_step = False,
            on_epoch = True,
            prog_bar = True,
        )
        return loss

 