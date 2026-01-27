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
            # 一般的なキー名: 'features', 'embeddings', 'coords' など
            # 実際のキー名を確認してください
            if 'features' in f.keys():
                embeddings = f['features'][:]
            elif 'embeddings' in f.keys():
                embeddings = f['embeddings'][:]
            else:
                # 最初のデータセットを使用
                key = list(f.keys())[0]
                embeddings = f[key][:]
            
            # numpy配列をPyTorchテンソルに変換
            embeddings = torch.from_numpy(embeddings).float()
        
        return embeddings, label

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
