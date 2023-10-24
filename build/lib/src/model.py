from typing import Dict, List, Optional
import pytorch_lightning as pl
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, n_features):
        super(Block, self).__init__()
        self.Fx = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=1),
        )

    def forward(self, x):
        fx = self.Fx(x)
        x = nn.ReLU()(fx + x)
        return x


class BatchBlock(nn.Module):
    def __init__(self, n_features):
        super(BatchBlock, self).__init__()
        self.Fx = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(n_features),
            nn.Conv2d(n_features, n_features, kernel_size=3, padding=1),
        )

        self.bat = nn.BatchNorm2d(n_features)

    def forward(self, x):
        fx = self.Fx(x)
        x = nn.ReLU()(fx + x)
        x = self.bat(x)
        return x


class Model(pl.LightningModule):
    def __init__(
        self,
        num_blocks: int = 2,
        num_features: int = 64,
        lr: Optional[float] = 1e-3,
        weight_decay: Optional[float] = 0,
        batch_size: Optional[int] = 1,
        batch_normalization: Optional[bool] = False,
        optimizer: Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        super(Model, self).__init__(*args, **kwargs)

        conv_layers = [
            nn.Conv2d(3, num_features, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        ]
        if batch_normalization:
            for i in range(num_blocks):
                conv_layers.append(BatchBlock(num_features))
        else:
            for i in range(num_blocks):
                conv_layers.append(Block(num_features))

        self.blocks = nn.Sequential(*conv_layers)

        self.rd = nn.MaxPool2d(2)

        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        self.lr = lr
        self.batch_size = batch_size
        self.loss = torch.nn.BCELoss()
        if optimizer is None or optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=weight_decay
            )
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x: List[str]) -> List[str]:
        """
        https://huggingface.co/docs/transformers/model_doc/t5#inference
        """

        x = self.blocks(x)

        x = self.rd(x)
        # reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

    def _inference_training(
        self, batch, batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        From https://huggingface.co/docs/transformers/model_doc/t5#training
        """

        data, target = batch
        output = self(data)
        output, target = output.flatten(), target.flatten()
        preds = torch.round(output)
        accuracy = self.accuracy(preds, target)
        target = target.to(torch.float32)

        return self.loss(output, target), accuracy

    def training_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy = self._inference_training(batch, batch_idx)
        self.log("train loss", loss, batch_size=self.batch_size)
        self.log("train accuracy", accuracy, batch_size=self.batch_size)
        return loss

    def validation_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy = self._inference_training(batch, batch_idx)
        self.log("val loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("val accuracy", accuracy, batch_size=self.batch_size, sync_dist=True)
        return loss

    def test_step(
        self, batch: List[str], batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        loss, accuracy = self._inference_training(batch, batch_idx)
        self.log("test loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("test accuracy", accuracy, batch_size=self.batch_size, sync_dist=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer

    def accuracy(self, preds, target):
        return (preds == target).sum() / len(target)
