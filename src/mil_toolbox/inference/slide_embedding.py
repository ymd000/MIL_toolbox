"""Slide-level embedding computation from patch embeddings and attention."""

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .predictor import MILPredictor


class SlideEmbeddingCalculator:
    """Compute slide-level embeddings from patch embeddings using attention weights.

    The slide embedding is computed as weighted sum of patch embeddings:
        slide_embedding = sum(attention_i * patch_embedding_i)

    Usage:
        calculator = SlideEmbeddingCalculator(
            model_class=MILModel,
            model_kwargs={"num_classes": 2, "model_config": "abmil.base.gigapath.none"},
            output_dir="./outputs"
        )
        calculator.load_models()

        # Single sample
        slide_emb = calculator.compute(patch_embeddings, fold_idx=0)

        # Batch processing
        embeddings, labels = calculator.compute_dataset(dataset, use_val_fold=True)
    """

    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        output_dir: str | Path,
        device: str = "auto",
    ):
        """Initialize the calculator.

        Args:
            model_class: Model class (e.g., MILModel)
            model_kwargs: Keyword arguments for model instantiation
            output_dir: Directory containing fold checkpoints
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.predictor = MILPredictor(
            model_class=model_class,
            model_kwargs=model_kwargs,
            output_dir=output_dir,
            device=device,
        )

    def load_models(self, checkpoint_name: str = "best") -> None:
        """Load models from all folds.

        Args:
            checkpoint_name: Checkpoint name to load ("best" or "last")
        """
        self.predictor.load_models(checkpoint_name)

    def compute(
        self,
        patch_embeddings: torch.Tensor,
        fold_idx: int,
        normalize: bool = False,
    ) -> dict:
        """Compute slide embedding for a single sample.

        Args:
            patch_embeddings: Tensor of shape (n_patches, embed_dim)
            fold_idx: Fold index to use for attention computation
            normalize: Whether to L2-normalize the output embedding

        Returns:
            dict with keys:
                - slide_embedding: Slide-level embedding (embed_dim,)
                - attention: Attention weights (n_patches,)
                - probs: Prediction probabilities
                - pred_class: Predicted class
        """
        result = self.predictor.predict(patch_embeddings, fold_idx, return_attention=True)

        attention = result["attention"]
        if attention is None:
            raise ValueError("Model did not return attention weights")

        # Ensure attention is 1D
        attention = attention.flatten()

        # Compute weighted sum of patch embeddings
        if patch_embeddings.dim() == 3:
            patch_embeddings = patch_embeddings.squeeze(0)

        slide_embedding = torch.matmul(attention, patch_embeddings.cpu())

        if normalize:
            slide_embedding = slide_embedding / slide_embedding.norm()

        return {
            "slide_embedding": slide_embedding,
            "attention": attention,
            "probs": result["probs"],
            "pred_class": result["pred_class"],
        }

    def compute_ensemble(
        self,
        patch_embeddings: torch.Tensor,
        normalize: bool = False,
        aggregation: str = "mean",
    ) -> dict:
        """Compute slide embedding using ensemble of all fold models.

        Args:
            patch_embeddings: Tensor of shape (n_patches, embed_dim)
            normalize: Whether to L2-normalize the output embedding
            aggregation: How to aggregate ("mean" attention or per-fold embeddings)

        Returns:
            dict with slide_embedding, attention, probs, pred_class
        """
        result = self.predictor.predict_ensemble(
            patch_embeddings,
            return_attention=True,
            aggregation=aggregation,
        )

        attention = result["attention"]
        if attention is None:
            raise ValueError("Models did not return attention weights")

        attention = attention.flatten()

        if patch_embeddings.dim() == 3:
            patch_embeddings = patch_embeddings.squeeze(0)

        slide_embedding = torch.matmul(attention, patch_embeddings.cpu())

        if normalize:
            slide_embedding = slide_embedding / slide_embedding.norm()

        return {
            "slide_embedding": slide_embedding,
            "attention": attention,
            "probs": result["probs"],
            "pred_class": result["pred_class"],
        }

    def compute_dataset(
        self,
        dataset,
        use_val_fold: bool = True,
        normalize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute slide embeddings for entire dataset.

        When use_val_fold=True, each sample is processed using the model
        from the fold where it was in the validation set (prevents data leakage).

        Args:
            dataset: Dataset with __len__ and __getitem__ returning (x, label)
            use_val_fold: Use validation fold model for each sample
            normalize: Whether to L2-normalize embeddings

        Returns:
            Tuple of (embeddings, labels) as numpy arrays
                - embeddings: shape (n_samples, embed_dim)
                - labels: shape (n_samples,)
        """
        embeddings = []
        labels = []

        if use_val_fold:
            # Process each fold's validation samples
            for fold_idx in range(self.predictor.num_folds):
                _, val_indices = self.predictor.get_fold_indices(fold_idx)

                for idx in val_indices:
                    x, label = dataset[idx]
                    result = self.compute(x, fold_idx, normalize=normalize)
                    embeddings.append(result["slide_embedding"].numpy())
                    labels.append(label)
        else:
            # Use fold 0 for all samples
            for idx in range(len(dataset)):
                x, label = dataset[idx]
                result = self.compute(x, fold_idx=0, normalize=normalize)
                embeddings.append(result["slide_embedding"].numpy())
                labels.append(label)

        return np.array(embeddings), np.array(labels)

    def compute_with_predictions(
        self,
        dataset,
        use_val_fold: bool = True,
        normalize: bool = False,
    ) -> dict:
        """Compute slide embeddings with full prediction results.

        Args:
            dataset: Dataset with __len__ and __getitem__ returning (x, label)
            use_val_fold: Use validation fold model for each sample
            normalize: Whether to L2-normalize embeddings

        Returns:
            dict with keys:
                - embeddings: np.ndarray of shape (n_samples, embed_dim)
                - labels: np.ndarray of shape (n_samples,)
                - predictions: np.ndarray of predicted classes
                - probabilities: np.ndarray of shape (n_samples, n_classes)
                - indices: list of sample indices in order processed
        """
        embeddings = []
        labels = []
        predictions = []
        probabilities = []
        indices = []

        if use_val_fold:
            for fold_idx in range(self.predictor.num_folds):
                _, val_indices = self.predictor.get_fold_indices(fold_idx)

                for idx in val_indices:
                    x, label = dataset[idx]
                    result = self.compute(x, fold_idx, normalize=normalize)

                    embeddings.append(result["slide_embedding"].numpy())
                    labels.append(label)
                    predictions.append(result["pred_class"])
                    probabilities.append(result["probs"].numpy())
                    indices.append(idx)
        else:
            for idx in range(len(dataset)):
                x, label = dataset[idx]
                result = self.compute(x, fold_idx=0, normalize=normalize)

                embeddings.append(result["slide_embedding"].numpy())
                labels.append(label)
                predictions.append(result["pred_class"])
                probabilities.append(result["probs"].numpy())
                indices.append(idx)

        return {
            "embeddings": np.array(embeddings),
            "labels": np.array(labels),
            "predictions": np.array(predictions),
            "probabilities": np.array(probabilities),
            "indices": indices,
        }
