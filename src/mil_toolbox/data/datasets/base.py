import torch
from torch.utils.data import Dataset


class WSIDataset(Dataset):
    def __init__(self, embeddings_list, labels):
        self.embeddings_list = embeddings_list
        self.labels = labels

    def __len__(self):
        return len(self.embeddings_list)

    def __getitem__(self, idx):
        return self.embeddings_list[idx], self.labels[idx]
