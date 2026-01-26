"""Fold管理: train/inferenceで共通して使用"""
import os
import csv
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold


@dataclass
class FoldInfo:
    """単一foldの情報"""
    fold_idx: int
    train_indices: List[int]
    val_indices: List[int]
    train_labels: List[int]
    val_labels: List[int]


class FoldManager:
    """Cross-validation用のfold管理クラス"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.folds: List[FoldInfo] = []
        self.num_folds: int = 0

    def create_folds(
        self,
        dataset,
        num_folds: int,
        shuffle: bool = True,
        random_state: int = 42
    ):
        """データセットからfoldを作成"""
        self.num_folds = num_folds
        labels = dataset.labels

        skf = StratifiedKFold(
            n_splits=num_folds,
            shuffle=shuffle,
            random_state=random_state
        )

        self.folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
            fold_info = FoldInfo(
                fold_idx=fold_idx,
                train_indices=train_idx.tolist(),
                val_indices=val_idx.tolist(),
                train_labels=[labels[i] for i in train_idx],
                val_labels=[labels[i] for i in val_idx]
            )
            self.folds.append(fold_info)

        return self.folds

    def save(self):
        """fold情報をCSVに保存"""
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = self.output_dir / "fold_indices.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "split", "indices", "labels"])

            for fold in self.folds:
                train_indices_str = ",".join(map(str, fold.train_indices))
                val_indices_str = ",".join(map(str, fold.val_indices))
                train_labels_str = ",".join(map(str, fold.train_labels))
                val_labels_str = ",".join(map(str, fold.val_labels))
                writer.writerow([fold.fold_idx, "train", train_indices_str, train_labels_str])
                writer.writerow([fold.fold_idx, "val", val_indices_str, val_labels_str])

        print(f"Fold indices saved to: {csv_path}")

    def load(self):
        """CSVからfold情報を読み込み"""
        csv_path = self.output_dir / "fold_indices.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Fold indices file not found: {csv_path}")

        fold_data = {}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fold_idx = int(row["fold"])
                split = row["split"]
                indices = [int(x) for x in row["indices"].split(",")]
                labels = [int(x) for x in row["labels"].split(",")]

                if fold_idx not in fold_data:
                    fold_data[fold_idx] = {}
                fold_data[fold_idx][split] = {"indices": indices, "labels": labels}

        self.folds = []
        for fold_idx in sorted(fold_data.keys()):
            data = fold_data[fold_idx]
            fold_info = FoldInfo(
                fold_idx=fold_idx,
                train_indices=data["train"]["indices"],
                val_indices=data["val"]["indices"],
                train_labels=data["train"]["labels"],
                val_labels=data["val"]["labels"]
            )
            self.folds.append(fold_info)

        self.num_folds = len(self.folds)
        print(f"Loaded {self.num_folds} folds from: {csv_path}")
        return self.folds

    def get_fold(self, fold_idx: int) -> FoldInfo:
        """指定foldの情報を取得"""
        return self.folds[fold_idx]

    def get_fold_dir(self, fold_idx: int) -> Path:
        """fold用ディレクトリパスを取得"""
        return self.output_dir / f"fold_{fold_idx}"

    def get_checkpoint_path(self, fold_idx: int, name: str = "best") -> Path:
        """チェックポイントパスを取得"""
        return self.get_fold_dir(fold_idx) / "checkpoints" / f"{name}.ckpt"

    def get_all_checkpoint_paths(self, name: str = "best") -> List[Path]:
        """全foldのチェックポイントパスを取得"""
        return [self.get_checkpoint_path(i, name) for i in range(self.num_folds)]
