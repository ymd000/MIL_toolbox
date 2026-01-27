import h5py
import torch
from torch.utils.data import Dataset
import os
import numpy as np

class WSIDataset(Dataset):
    def __init__(self, data_dir='data/embedding', label_dict={}):
        """
        Args:
            data_dir: HDF5ファイルが格納されているディレクトリ
            label_dict: ファイル名(拡張子なし)をキー、ラベルを値とする辞書
        """
        self.data_dir = data_dir
        self.h5_files = []
        self.labels = []
        
        # HDF5ファイルのリストを取得
        for file in sorted(os.listdir(data_dir)):
            if file.endswith('.h5'):
                file_path = os.path.join(data_dir, file)
                file_name = os.path.splitext(file)[0]
                self.h5_files.append(file_path)

                # label_dictからラベルを取得（存在しない場合は-1）
                label = label_dict.get(file_name, -1)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.h5_files)
    
    def __getitem__(self, idx):
        """
        HDF5ファイルから埋め込みデータを読み込む
        
        Returns:
            embeddings: torch.Tensor of shape (num_patches, feature_dim)
            label: int
        """
        h5_path = self.h5_files[idx]
        label = self.labels[idx]
        
        with h5py.File(h5_path, 'r') as f:
            # HDF5ファイルの構造に応じて適切なキーを指定
            embeddings = f['features'][:]
            embeddings = torch.from_numpy(embeddings).float()
        
        return embeddings, label

