import torch
from torch.nn.utils.rnn import pad_sequence


def mil_collate_fn(batch: list[tuple[torch.Tensor, int]]):
    """Variable-length WSI features をパディングしてバッチ化する。

    Args:
        batch: [(features: Tensor(N_i, D), label: int), ...]

    Returns:
        features: Tensor(B, max_N, D)  パディング済み特徴量
        mask:     Tensor(B, max_N)     1=実パッチ / 0=パディング
        labels:   Tensor(B,)
    """
    features_list, labels = zip(*batch)

    # (B, max_N, D) にパディング（デフォルトでゼロ埋め）
    padded = pad_sequence(features_list, batch_first=True)  # (B, max_N, D)

    # マスク作成: 実パッチ位置を 1、パディング位置を 0
    max_n = padded.size(1)
    mask = torch.zeros(len(features_list), max_n, dtype=torch.float32)
    for i, feat in enumerate(features_list):
        mask[i, :feat.size(0)] = 1.0

    labels = torch.tensor(labels, dtype=torch.long)

    return padded, mask, labels
