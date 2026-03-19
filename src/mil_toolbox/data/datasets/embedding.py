"""埋め込みベクトルを保持するシンプルなデータセット"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """スライド埋め込みベクトルとラベルを保持するデータセット。

    Args:
        embeddings: 形状 (N, D) の埋め込みベクトル。np.ndarray / Tensor / list を受け付ける。
        labels: 長さ N のラベル列。np.ndarray / Tensor / list を受け付ける。
        case_names: 長さ N の症例名リスト（省略可）。
    """

    def __init__(
        self,
        embeddings,
        labels,
        case_names: list[str] | None = None,
    ):
        if not isinstance(embeddings, Tensor):
            embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
        else:
            embeddings = embeddings.float()

        if not isinstance(labels, Tensor):
            labels = torch.tensor(np.array(labels), dtype=torch.int64)
        else:
            labels = labels.long()

        assert len(embeddings) == len(labels), (
            f"embeddings と labels の長さが一致しません: {len(embeddings)} != {len(labels)}"
        )

        self._embeddings = embeddings
        self._labels = labels
        self.case_names = case_names

    @classmethod
    def from_results(cls, results: dict, reindex: bool = True) -> "EmbeddingDataset":
        """SlideEmbeddingCalculator の results dict から構築するクラスメソッド。

        Args:
            results: ``compute_and_save`` 等が返す dict。
                必須キー: ``embeddings``, ``labels``。
                任意キー: ``indices`` (use_val_fold=True のとき fold 順に返るため並べ直しに使用),
                          ``case_names``。
            reindex: True のとき ``results["indices"]`` が存在すれば元のデータセット順に並べ直す。
        """
        embeddings = results["embeddings"]
        labels = results["labels"]
        case_names = results.get("case_names", None)

        if reindex and "indices" in results and results["indices"] is not None:
            indices = results["indices"]
            order = np.argsort(indices)
            embeddings = embeddings[order]
            labels = labels[order]
            if case_names is not None:
                case_names = [case_names[i] for i in order]

        return cls(embeddings, labels, case_names)

    def __len__(self) -> int:
        return len(self._embeddings)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return self._embeddings[idx], int(self._labels[idx])

    @property
    def embedding_dim(self) -> int:
        """埋め込みベクトルの次元数"""
        return self._embeddings.shape[1]

    @property
    def num_classes(self) -> int:
        """クラス数（ラベルのユニーク数）"""
        return int(self._labels.max().item()) + 1

    @property
    def labels(self) -> list[int]:
        """ラベルのリスト（FoldManager.create_folds との互換用）"""
        return self._labels.tolist()
