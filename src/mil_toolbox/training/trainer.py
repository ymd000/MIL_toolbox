import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import torch.nn as nn
import numpy as np
from mil_lab.builder import create_model
from sklearn.model_selection import KFold

class WSIDataset(Dataset):
    def __init__(self, embeddings_list, labels):
        self.embeddings_list = embeddings_list
        self.labels = labels
    
    def __len__(self):
        return len(self.embeddings_list)
    
    def __getitem__(self, idx):
        return self.embeddings_list[idx], self.labels[idx]
        
class MILDataModule(L.LightningDataModule):
    def __init__(
            self,
            batch_size: int = 1,
            num_wsi: int = 10,
            embedding_dim: int = 1024,
            num_workers: int = 4
        ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_test_wsi = int(num_wsi / 5)
        self.num_train_wsi = num_wsi - self.num_test_wsi
        self.embedding_dim = embedding_dim
        self.num_workers = num_workers
        
    def prepare_data(self) -> None:
        self.train_embeddings = [
            torch.randn(torch.randint(50, 501, (1,)).item(), self.embedding_dim)
            for i in range(self.num_train_wsi)
        ]
        self.train_labels = torch.randint(0, 2, (self.num_train_wsi,))
        self.test_embeddings = [
            torch.randn(torch.randint(50, 501, (1,)).item(), self.embedding_dim)
            for i in range(self.num_test_wsi)
        ]
        self.test_labels = torch.randint(0, 2, (self.num_test_wsi,)) 

    def setup(self, stage=None):
        self.train_dataset = WSIDataset(self.train_embeddings, self.train_labels)
        self.test_dataset = WSIDataset(self.test_embeddings, self.test_labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
class MILmodel(L.LightningModule):
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

    def forward(self, x):
        results, _ = self.model(x)
        logits = results['logits']
        return logits

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch):
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer








class DummyWSIDataset(Dataset):
    def __init__(self, num_wsi=10):
        self.data = []
        self.labels = []
        for i in range(num_wsi):
            num_patches = np.random.randint(50, 501)
            wsi = torch.randn(num_patches, 1024)
            label = np.random.randint(0, 2)
            
            self.data.append(wsi)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



class CrossValidationTrainer:
    def __init__(
        self,
        dataset,
        num_fold: int,
        shuffle: bool = True,
        random_state: int = 42
    ):
        self.dataset = dataset
        self.num_fold = num_fold
        self.shuffle = shuffle
        self.random_state = random_state

    def k_spit(self):
        self.k_fold_dataset = KFold(
            n_splits=self.num_fold,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

    def run(self):
        self.k_spit()
        results = []
        for fold_idx, (train_idx, val_idx) in enumerate(self.k_fold_dataset.split(self.dataset)):
            print(f"\n=== Fold {fold_idx + 1}/{self.num_fold} ===")
            fold_result = self.run_one_fold(fold_idx, train_idx, val_idx)
            results.append(fold_result)
        return results
        
    def run_one_fold(self, fold_idx: int, train_idx, val_idx):
        train_data = torch.utils.data.Subset(self.dataset, train_idx)
        val_data = torch.utils.data.Subset(self.dataset, val_idx)   
    
        print(f"Train WSIs: {len(train_data)}, Val WSIs: {len(val_data)}")
        print(f"Train indices: {train_idx}")
        print(f"Val indices: {val_idx}")
    
        train_patches = []
        train_labels = []
        for i in range(len(train_data)):
            wsi, label = train_data[i]
            train_patches.append(wsi.shape[0])
            train_labels.append(label)
    
        val_patches = []
        val_labels = []
        for i in range(len(val_data)):
            wsi, label = val_data[i]
            val_patches.append(wsi.shape[0])
            val_labels.append(label)
    
        print(f"Train patches per WSI: {train_patches}")
        print(f"Train labels: {train_labels} (class 0: {train_labels.count(0)}, class 1: {train_labels.count(1)})")
        print(f"Val patches per WSI: {val_patches}")
        print(f"Val labels: {val_labels} (class 0: {val_labels.count(0)}, class 1: {val_labels.count(1)})")
    
        return {
            'fold': fold_idx,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'train_patches': train_patches,
            'val_patches': val_patches,
            'train_labels': train_labels,
            'val_labels': val_labels
        }

def main():
    """
    model_name = "abmil"
    model_config = f'{model_name}.base.uni.none'

    datamodule = MILDataModule(
            )

    model = MILmodel(
                num_classes=2,
                model_config=model_config
            )

    trainer = L.Trainer(
                max_epochs=5,
                accelerator='auto',
                devices=1,
                log_every_n_steps=1
            )    
    trainer.fit(
                model=model,
                datamodule=datamodule
            )
    """

    dataset = DummyWSIDataset(num_wsi=10)
    
    print("Dataset created:")
    for i in range(len(dataset)):
        wsi, label = dataset[i]
        print(f"WSI {i}: {wsi.shape}, Label {i}: {label}")
    
    # CrossValidation実行
    trainer = CrossValidationTrainer(dataset, num_fold=5)
    results = trainer.run()
    
    print("\n=== Summary ===")
    for result in results:
        print(f"Fold {result['fold']}:")
        print(f"Train={result['train_size']} (0:{result['train_labels'].count(0)}, 1:{result['train_labels'].count(1)})")
        print(f"Val={result['val_size']} (0:{result['val_labels'].count(0)}, 1:{result['val_labels'].count(1)})")

    
if __name__ == "__main__":
    main()
