import csv
import warnings
import h5py
import psutil
import torch
from torch.utils.data import Dataset
from pathlib import Path


FEATURE_DIMS = {
    "uni": 1024,
    "uni2": 1536,
    "gigapath": 1536,
    "virchow2": 2560,
    "conch15": 768,
}

class WSIDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        model: str,
        csv_path: str,
        use_cache: bool = True,
    ):
        """
        HDF5ファイルからWSI embeddingを読み込むデータセット

        Args:
            data_dir: HDF5ファイルが格納されているディレクトリ
            model: モデル名 (uni, gigapath, virchow2)
            csv_path: ラベルCSVファイルのパス (slide_id, label)
            use_cache: Trueのとき、読み込んだembeddingをメモリにキャッシュする (default: True)
        """
        self.data_dir = Path(data_dir)
        self.model = model
        self.feature_dim = FEATURE_DIMS.get(model)
        self.use_cache = use_cache
        self._cache: dict[int, torch.Tensor] = {}
        self._memory_warned = False

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

    def _load_embeddings(self, idx: int) -> torch.Tensor:
        if self.use_cache and idx in self._cache:
            return self._cache[idx]

        with h5py.File(self.h5_files[idx], 'r') as f:
            embeddings = torch.from_numpy(f[f'{self.model}/features'][:]).float()

        if self.use_cache:
            self._cache[idx] = embeddings
            if not self._memory_warned:
                mem = psutil.virtual_memory()
                if mem.percent >= 90.0:
                    warnings.warn(
                        f"RAM usage is {mem.percent:.1f}% "
                        f"({mem.available / 1e9:.1f} GB available). "
                        "Consider disabling cache (use_cache=False).",
                        ResourceWarning,
                        stacklevel=2,
                    )
                    self._memory_warned = True

        return embeddings

    def __getitem__(self, idx):
        """
        Returns:
            embeddings: torch.Tensor of shape (N, D)
            label: int
        """
        return self._load_embeddings(idx), self.labels[idx]
