import torch
import torch.nn as nn
import lightning as L
from mil_lab.builder import create_model


class MILModel(L.LightningModule):
    def __init__(
            self,
            num_classes: int,
            model_config: str,
            **model_kwargs
        ):
        super().__init__()

        self.model = create_model(
                        model_config,
                        pretrained=False,
                        num_classes=num_classes,
                        **model_kwargs
                    )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, return_attention: bool = True):
        results, log_dict = self.model(x, return_attention=return_attention)
        logits = results['logits']
        attention = log_dict.get('attention', None) if return_attention else None
        return {'logits': logits, 'attention': attention}

    def training_step(self, batch):
        x, y = batch
        outputs = self(x)
        logits = outputs['logits']
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch):
        x, y = batch
        outputs = self(x)
        logits = outputs['logits']
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
