import torch
from torch.utils.data import DataLoader
import lightning as L

from .datasets import WSIDataset


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
