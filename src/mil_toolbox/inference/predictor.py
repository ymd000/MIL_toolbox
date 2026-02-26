"""MIL model predictor for cross-validation trained models."""

from pathlib import Path
from typing import Optional

import torch

from mil_toolbox.data import FoldManager


class MILPredictor:
    """Cross-validation で学習した複数モデルを使った推論クラス.

    Usage:
        predictor = MILPredictor(
            model_class=MILModel,
            model_kwargs={"num_classes": 2, "model_config": "abmil.base.gigapath.none"},
            output_dir="./outputs"
        )
        predictor.load_models(checkpoint_name="best")

        # Single fold prediction
        result = predictor.predict(x, fold_idx=0)

        # Ensemble prediction (average across all folds)
        result = predictor.predict_ensemble(x)
    """

    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        output_dir: str | Path,
        device: str = "auto",
    ):
        """Initialize the predictor.

        Args:
            model_class: Model class (e.g., MILModel)
            model_kwargs: Keyword arguments for model instantiation
            output_dir: Directory containing fold checkpoints
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.output_dir = Path(output_dir)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.fold_manager = FoldManager(str(self.output_dir))
        self.models: list = []

    def load_models(self, checkpoint_name: str = "best") -> None:
        """Load models from all folds.

        Args:
            checkpoint_name: Checkpoint name to load ("best" or "last")
        """
        self.fold_manager.load()

        self.models = []
        for fold_idx in range(self.fold_manager.num_folds):
            ckpt_path = self.fold_manager.get_checkpoint_path(fold_idx, checkpoint_name)
            model = self._load_single_model(ckpt_path)
            self.models.append(model)
            print(f"Loaded model for fold {fold_idx}: {ckpt_path}")

    def _load_single_model(self, checkpoint_path: str | Path):
        """Load a single model from checkpoint."""
        model = self.model_class.load_from_checkpoint(
            str(checkpoint_path),
            **self.model_kwargs,
        )
        model.to(self.device)
        model.eval()
        return model

    def predict(
        self,
        x: torch.Tensor,
        fold_idx: int,
        return_attention: bool = True,
    ) -> dict:
        """Predict using a specific fold model.

        Args:
            x: Input tensor of shape (n_patches, embed_dim) or (1, n_patches, embed_dim)
            fold_idx: Fold index to use for prediction
            return_attention: Whether to return attention weights

        Returns:
            dict with keys:
                - logits: Raw model outputs
                - probs: Softmax probabilities
                - pred_class: Predicted class index
                - attention: Attention weights (if return_attention=True)
        """
        if fold_idx >= len(self.models):
            raise ValueError(f"Fold index {fold_idx} out of range (loaded {len(self.models)} models)")

        return self._predict_with_model(x, self.models[fold_idx], return_attention)

    def predict_ensemble(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        aggregation: str = "mean",
    ) -> dict:
        """Predict using ensemble of all fold models.

        Args:
            x: Input tensor of shape (n_patches, embed_dim) or (1, n_patches, embed_dim)
            return_attention: Whether to return attention weights
            aggregation: How to aggregate predictions ("mean" or "vote")

        Returns:
            dict with keys:
                - logits: Aggregated logits
                - probs: Aggregated probabilities
                - pred_class: Predicted class
                - attention: Aggregated attention weights (if return_attention=True)
                - fold_predictions: List of individual fold predictions
        """
        fold_results = []
        for fold_idx in range(len(self.models)):
            result = self.predict(x, fold_idx, return_attention)
            fold_results.append(result)

        # Aggregate probabilities
        all_probs = torch.stack([r["probs"] for r in fold_results])
        all_logits = torch.stack([r["logits"] for r in fold_results])

        if aggregation == "mean":
            avg_probs = all_probs.mean(dim=0)
            avg_logits = all_logits.mean(dim=0)
            pred_class = avg_probs.argmax().item()
        elif aggregation == "vote":
            votes = torch.stack([r["probs"].argmax() for r in fold_results])
            pred_class = votes.mode().values.item()
            avg_probs = all_probs.mean(dim=0)
            avg_logits = all_logits.mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        result = {
            "logits": avg_logits,
            "probs": avg_probs,
            "pred_class": pred_class,
            "fold_predictions": fold_results,
        }

        if return_attention:
            all_attention = [r["attention"] for r in fold_results if r["attention"] is not None]
            if all_attention:
                result["attention"] = torch.stack(all_attention).mean(dim=0)
            else:
                result["attention"] = None

        return result

    def _predict_with_model(
        self,
        x: torch.Tensor,
        model,
        return_attention: bool = True,
    ) -> dict:
        """Run prediction with a single model."""
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        with torch.no_grad():
            outputs = model(x)
            logits = outputs["logits"].squeeze()
            probs = torch.softmax(logits, dim=-1)
            pred_class = probs.argmax().item()

            attention = None
            if return_attention and "attention" in outputs:
                attention = outputs["attention"].squeeze()

        result = {
            "logits": logits.cpu(),
            "probs": probs.cpu(),
            "pred_class": pred_class,
        }

        if return_attention:
            result["attention"] = attention.cpu() if attention is not None else None

        return result

    @property
    def num_folds(self) -> int:
        """Return the number of loaded folds."""
        return len(self.models)

    def get_fold_indices(self, fold_idx: int) -> tuple[list[int], list[int]]:
        """Get train and validation indices for a specific fold.

        Args:
            fold_idx: Fold index

        Returns:
            Tuple of (train_indices, val_indices)
        """
        fold_info = self.fold_manager.get_fold(fold_idx)
        return fold_info.train_indices, fold_info.val_indices
