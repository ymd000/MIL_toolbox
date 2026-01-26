import torch
from torch.utils.data import Dataset
import numpy as np


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
