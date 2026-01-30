import torch
from typing import List, Optional

import umap

from mil_toolbox.data import FoldManager


class AttentionAggregator:
    """Cross-validation で学習した複数モデルからattentionを集約するクラス"""

    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        output_dir: str,
        device: str = "auto"
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # FoldManager初期化
        self.fold_manager = FoldManager(output_dir)
        self.models: List = []

    def load_models(self, checkpoint_name: str = "best"):
        """FoldManagerからチェックポイントをロード"""
        self.fold_manager.load()

        self.models = []
        for fold_idx in range(self.fold_manager.num_folds):
            ckpt_path = self.fold_manager.get_checkpoint_path(fold_idx, checkpoint_name)
            model = self._load_model(ckpt_path)
            self.models.append(model)
            print(f"Loaded model for fold {fold_idx}: {ckpt_path}")

    def _load_model(self, checkpoint_path):
        model = self.model_class.load_from_checkpoint(
            checkpoint_path,
            **self.model_kwargs
        )
        model.to(self.device)
        model.eval()
        return model

    def predict_with_attention(self, x: torch.Tensor, fold_idx: Optional[int]):
        """単一サンプルに対してlogitsとattentionを取得"""
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        return self._predict_single_model(x, self.models[fold_idx])

    def _predict_single_model(self, x: torch.Tensor, model):
        with torch.no_grad():
            outputs = model(x)
            logits = outputs['logits']
            attention = outputs.get('attention', None).squeeze()
            probs = torch.softmax(logits, dim=-1).squeeze()

            patch_embeddings = x.squeeze(0)
            slide_embedding = torch.matmul(attention, patch_embeddings)

        return {
            'logits': logits.cpu(),
            'attention': attention.cpu() if attention is not None else None,
            'probs': probs.cpu(),
            'slide_embedding': slide_embedding.cpu()
        }
