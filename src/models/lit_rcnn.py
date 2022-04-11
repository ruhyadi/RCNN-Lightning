import torch
from torch import nn
import pytorch_lightning as pl

from typing import Any, List

class LitWheat(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module = None,
        lr: float = 0.001,
        weight_decay: float = 0.001
    ):
        super(LitWheat, self).__init__()
        self.save_hyperparameters(logger=False)

        self.net = net

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        return [optimizer]

    def step(self, batch: Any):
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        return images, targets

    def train_step(self, batch, batch_idx):
        images, targets = self.step(batch)

        # separate losses
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # logger
        self.log("train/loss", losses, on_step=True, on_epoch=True, prog_bar=True)

        return {'loss': losses}

    def validation_step(self, batch, batch_idx):
        images, targets = self.step(batch)

        # separate losses
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # logger
        self.log("val/loss", losses, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': losses}