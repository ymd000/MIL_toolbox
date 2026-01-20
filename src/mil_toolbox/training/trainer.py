import torch
import lightning as L
import torch.nn as nn
from mil_lab.builder import create_model

model_name = "abmil"
model_config = f'{model_name}'

test_embedding = torch.rand(100, 1024)
test_embedding = test_embedding.unsqueeze(0)

class MILtrainer(L.LightningModule):
    def __init__(
            self,
            num_classes: int,
            model_config: str = model_config,
            **model_kwargs
        ):
        
        super().__init__()
        
        # Create model        
        self.model = create_model(model_config, num_classes=num_classes, **model_kwargs)  
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        results, _ = self.model(x)
        logits = results['logits']
        return logits

    def training_step(self, batch, batch_idx):
        # It is independent of forward
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def main():
    trainer = MILtrainer(
                num_classes=2
            )
    print(trainer.model)
    logits = trainer(test_embedding)
    print(logits.shape)

if __name__ == "__main__":
    main()
