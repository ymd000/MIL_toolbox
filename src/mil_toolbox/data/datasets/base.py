import csv
import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path


FEATURE_DIMS = {
    "uni": 1024,
    "gigapath": 1536,
    "virchow2": 2560,
}


class WSIDataset(Dataset):
    def __init__(self, data_dir: str, model: str, csv_path: str):
        """
        HDF5ファイルからWSI embeddingを読み込むデータセット

        Args:
            data_dir: HDF5ファイルが格納されているディレクトリ
            model: モデル名 (uni, gigapath, virchow2)
            csv_path: ラベルCSVファイルのパス (slide_id, label)
        """
        self.data_dir = Path(data_dir)
        self.model = model
        self.feature_dim = FEATURE_DIMS.get(model)

        # CSVからラベル辞書を作成
        label_dict = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_dict[row['case_id']] = int(row['label'])

        self.h5_files = []
        self.labels = []

        for file in sorted(self.data_dir.iterdir()):
            if file.suffix == '.h5':
                file_name = file.stem
                self.h5_files.append(file)
                label = label_dict.get(file_name, -1)
                self.labels.append(label)

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        """
        Returns:
            embeddings: torch.Tensor of shape (N, D)
            label: int
        """
        h5_path = self.h5_files[idx]
        label = self.labels[idx]

        with h5py.File(h5_path, 'r') as f:
            embeddings = f[f'{self.model}/features'][:]
            embeddings = torch.from_numpy(embeddings).float()

        return embeddings, label
