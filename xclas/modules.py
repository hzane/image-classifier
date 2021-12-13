import torch
import torch.nn as nn
import torchmetrics.functional as M
import timm
from pytorch_lightning import LightningModule

from typing import Any


class LitClasModule(LightningModule):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = 'convit_small',
        lr: float = 1.e-3,
        scheduler_patience: int = 3,
        lr_reduce_factor: float = 0.33,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = self.configure_criterion()
        self.model = self.configure_model()

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

    def configure_model(self, ):
        model = timm.create_model(
            self.hparams.backbone_name,
            pretrained = True,
            num_classes = self.hparams.num_classes,
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

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)

        return dict(loss = loss, yhat = yhat, y = y)

    def validation_step_end(self, outputs):
        # yhat, y 会被自动reduce 所以增加这个空函数
        ...

    def validation_epoch_end(self, step_returns):
        if self.trainer.sanity_checking:
            return
        nc = self.hparams.num_classes

        yhat, y, loss = zip(
            *[(x['yhat'], x['y'], x['loss']) for x in step_returns]
        )
        yhat = torch.cat(yhat)
        y = torch.cat(y)
        loss = torch.cat(loss).mean()

        auroc = M.auroc(yhat, y, num_classes = nc)
        ap = M.average_precision(yhat, y, num_classes = nc)  # pr auc

        prec, recall = M.precision_recall(yhat, y)
        f1 = M.f1(yhat, y, num_classes = nc)
        acc = M.accuracy(yhat, y, num_classes = nc)

        self.log_dict(
            {
                'valid/auroc': auroc.item(),
                'valid/loss': loss.item(),
                'valid/F1': f1.item(),
                'valid/acc': acc.item(),
            },
            prog_bar = True,
        )
        self.log_dict({
            'valid/mAP': ap.item(),
            'valid/prec': prec.item(),
            'valid/recall': recall.item(),
        },
                      prog_bar = False)
        self.log('hp_metric', acc.item())
