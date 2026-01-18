import lightning as L
from mil_lab.builder import create_model

model_name = "abmil"
model_config = f'{model_name}'

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
        

def main():
    trainer = MILtrainer(
                num_classes=2
            )
    print(trainer.model)

if __name__ == "__main__":
    main()
