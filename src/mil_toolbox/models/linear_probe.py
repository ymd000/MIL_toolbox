"""線形プローブモデル（埋め込みベクトルの分類器）"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinearProbeModel(L.LightningModule):
    """埋め込みベクトルを入力とする線形分類器。

    MILPredictor._predict_with_model との互換のため、forward は
    ``{"logits": Tensor, "attention": None}`` を返す。

    Args:
        embedding_dim: 入力埋め込みの次元数。
        num_classes: 出力クラス数。
        lr: 学習率。
        dropout: Dropout 率（0.0 で無効）。
        weight_decay: Adam の weight_decay。
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        layers: list[nn.Module] = []
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(embedding_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> dict:
        """順伝播。

        Args:
            x: 形状 (B, D) の埋め込みテンソル。

        Returns:
            {"logits": Tensor(B, C), "attention": None}
        """
        logits = self.classifier(x)
        return {"logits": logits, "attention": None}

    def training_step(self, batch, batch_idx):
        x, y = batch  # (B, D), (B,)
        out = self(x)
        loss = F.cross_entropy(out["logits"], y)
        preds = out["logits"].argmax(dim=-1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out["logits"], y)
        preds = out["logits"].argmax(dim=-1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("val_acc", acc, sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
