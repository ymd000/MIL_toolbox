"""Slide-level embedding computation from patch embeddings and attention."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
        slide_emb = calculator.compute_abmil(patch_embeddings, fold_idx=0)

        # Batch processing
        embeddings, labels = calculator.compute_dataset(dataset, use_val_fold=True)
    """

    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        output_dir: str | Path,
        device: str = "auto",
        method_name: str | None = None,
    ):
        """Initialize the calculator.

        Args:
            model_class: Model class (e.g., MILModel)
            model_kwargs: Keyword arguments for model instantiation
            output_dir: Directory containing fold checkpoints
            device: Device to use ("auto", "cuda", "cpu")
            method_name: Name for saving slide embeddings (e.g., "abmil").
                         If None, extracted from model_config.
        """
        self.predictor = MILPredictor(
            model_class=model_class,
            model_kwargs=model_kwargs,
            output_dir=output_dir,
            device=device,
        )

        # Extract method name for HDF5 storage
        if method_name is not None:
            self.method_name = method_name
        elif "model_config" in model_kwargs:
            # Extract from model_config (e.g., "abmil.base.gigapath.none" -> "abmil")
            self.method_name = model_kwargs["model_config"].split(".")[0]
        else:
            self.method_name = "mil"

    def load_models(self, checkpoint_name: str = "best") -> None:
        """Load models from all folds.

        Args:
            checkpoint_name: Checkpoint name to load ("best" or "last")
        """
        self.predictor.load_models(checkpoint_name)

    def compute_abmil(
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

    def compute_nearest_cosine(
        self,
        patch_embeddings: torch.Tensor,
        normalize: bool = False,
    ) -> dict:
        """Compute slide embedding by selecting the patch nearest to mean (cosine similarity).

        No trained model is required.

        Args:
            patch_embeddings: Tensor of shape (n_patches, embed_dim)
            normalize: Whether to L2-normalize the output embedding

        Returns:
            dict with keys:
                - slide_embedding: Selected patch embedding (embed_dim,)
                - attention: None
                - probs: None
                - pred_class: None
                - selected_index: Index of the selected patch
        """
        if patch_embeddings.dim() == 3:
            patch_embeddings = patch_embeddings.squeeze(0)

        mean_emb = patch_embeddings.mean(dim=0)
        similarities = F.cosine_similarity(
            patch_embeddings, mean_emb.unsqueeze(0), dim=1
        )
        selected_index = similarities.argmax().item()
        slide_embedding = patch_embeddings[selected_index].clone()

        if normalize:
            slide_embedding = slide_embedding / slide_embedding.norm()

        return {
            "slide_embedding": slide_embedding,
            "attention": None,
            "probs": None,
            "pred_class": None,
            "selected_index": selected_index,
        }

    def compute_nearest_euclidean(
        self,
        patch_embeddings: torch.Tensor,
        normalize: bool = False,
    ) -> dict:
        """Compute slide embedding by selecting the patch nearest to mean (Euclidean distance).

        No trained model is required.

        Args:
            patch_embeddings: Tensor of shape (n_patches, embed_dim)
            normalize: Whether to L2-normalize the output embedding

        Returns:
            dict with keys:
                - slide_embedding: Selected patch embedding (embed_dim,)
                - attention: None
                - probs: None
                - pred_class: None
                - selected_index: Index of the selected patch
        """
        if patch_embeddings.dim() == 3:
            patch_embeddings = patch_embeddings.squeeze(0)

        mean_emb = patch_embeddings.mean(dim=0)
        distances = torch.cdist(
            mean_emb.unsqueeze(0), patch_embeddings
        ).squeeze(0)
        selected_index = distances.argmin().item()
        slide_embedding = patch_embeddings[selected_index].clone()

        if normalize:
            slide_embedding = slide_embedding / slide_embedding.norm()

        return {
            "slide_embedding": slide_embedding,
            "attention": None,
            "probs": None,
            "pred_class": None,
            "selected_index": selected_index,
        }

    def compute_top_attention(
        self,
        patch_embeddings: torch.Tensor,
        fold_idx: int,
        normalize: bool = False,
    ) -> dict:
        """Compute slide embedding using the patch with highest attention.

        Uses the trained model to compute attention, selects the patch
        with the highest attention weight.

        Args:
            patch_embeddings: Tensor of shape (n_patches, embed_dim)
            fold_idx: Fold index to use for attention computation
            normalize: Whether to L2-normalize the output embedding

        Returns:
            dict with keys:
                - slide_embedding: Top-attention patch embedding (embed_dim,)
                - attention: Full attention weights (n_patches,)
                - probs: None
                - pred_class: None
                - selected_index: Index of the selected patch
        """
        # Get attention from full bag
        result = self.predictor.predict(
            patch_embeddings, fold_idx, return_attention=True
        )
        attention = result["attention"]
        if attention is None:
            raise ValueError("Model did not return attention weights")
        attention = attention.flatten()

        if patch_embeddings.dim() == 3:
            patch_embeddings = patch_embeddings.squeeze(0)

        # Select patch with highest attention
        selected_index = attention.argmax().item()
        slide_embedding = patch_embeddings[selected_index].clone()

        if normalize:
            slide_embedding = slide_embedding / slide_embedding.norm()

        return {
            "slide_embedding": slide_embedding,
            "attention": attention,
            "probs": None,
            "pred_class": None,
            "selected_index": selected_index,
        }

    def compute_nearest_attention_cosine(
        self,
        patch_embeddings: torch.Tensor,
        fold_idx: int,
        normalize: bool = False,
    ) -> dict:
        """Compute slide embedding by finding patch nearest to attention-weighted embedding (cosine).

        Computes the attention-weighted sum embedding, then finds the single patch
        that has the highest cosine similarity to it.

        Args:
            patch_embeddings: Tensor of shape (n_patches, embed_dim)
            fold_idx: Fold index to use for attention computation
            normalize: Whether to L2-normalize the output embedding

        Returns:
            dict with keys:
                - slide_embedding: Selected patch embedding (embed_dim,)
                - attention: Full attention weights (n_patches,)
                - probs: None
                - pred_class: None
                - selected_index: Index of the selected patch
        """
        # Get attention-weighted embedding
        result = self.compute_abmil(patch_embeddings, fold_idx, normalize=False)
        attention_emb = result["slide_embedding"]
        attention = result["attention"]

        if patch_embeddings.dim() == 3:
            patch_embeddings = patch_embeddings.squeeze(0)

        # Find patch with highest cosine similarity to attention embedding
        similarities = F.cosine_similarity(
            patch_embeddings.cpu(), attention_emb.unsqueeze(0), dim=1
        )
        selected_index = similarities.argmax().item()
        slide_embedding = patch_embeddings[selected_index].clone()

        if normalize:
            slide_embedding = slide_embedding / slide_embedding.norm()

        return {
            "slide_embedding": slide_embedding,
            "attention": attention,
            "probs": None,
            "pred_class": None,
            "selected_index": selected_index,
        }

    def compute_nearest_attention_euclidean(
        self,
        patch_embeddings: torch.Tensor,
        fold_idx: int,
        normalize: bool = False,
    ) -> dict:
        """Compute slide embedding by finding patch nearest to attention-weighted embedding (Euclidean).

        Computes the attention-weighted sum embedding, then finds the single patch
        that has the smallest Euclidean distance to it.

        Args:
            patch_embeddings: Tensor of shape (n_patches, embed_dim)
            fold_idx: Fold index to use for attention computation
            normalize: Whether to L2-normalize the output embedding

        Returns:
            dict with keys:
                - slide_embedding: Selected patch embedding (embed_dim,)
                - attention: Full attention weights (n_patches,)
                - probs: None
                - pred_class: None
                - selected_index: Index of the selected patch
        """
        # Get attention-weighted embedding
        result = self.compute_abmil(patch_embeddings, fold_idx, normalize=False)
        attention_emb = result["slide_embedding"]
        attention = result["attention"]

        if patch_embeddings.dim() == 3:
            patch_embeddings = patch_embeddings.squeeze(0)

        # Find patch with smallest Euclidean distance to attention embedding
        distances = torch.cdist(
            attention_emb.unsqueeze(0), patch_embeddings.cpu()
        ).squeeze(0)
        selected_index = distances.argmin().item()
        slide_embedding = patch_embeddings[selected_index].clone()

        if normalize:
            slide_embedding = slide_embedding / slide_embedding.norm()

        return {
            "slide_embedding": slide_embedding,
            "attention": attention,
            "probs": None,
            "pred_class": None,
            "selected_index": selected_index,
        }

    def compute_attention_filtered_nearest_cosine(
        self,
        patch_embeddings: torch.Tensor,
        fold_idx: int,
        normalize: bool = False,
        threshold_quantile: float = 0.5,
    ) -> dict:
        """Compute slide embedding from attention-filtered patches, nearest to their mean (cosine).

        Selects patches with attention >= threshold_quantile quantile,
        computes their mean embedding, then returns the patch with the
        highest cosine similarity to that mean.

        Args:
            patch_embeddings: Tensor of shape (n_patches, embed_dim)
            fold_idx: Fold index to use for attention computation
            normalize: Whether to L2-normalize the output embedding
            threshold_quantile: Quantile threshold for attention filtering (default: 0.5 = median)

        Returns:
            dict with keys:
                - slide_embedding: Selected patch embedding (embed_dim,)
                - attention: Full attention weights (n_patches,)
                - probs: Prediction probabilities
                - pred_class: Predicted class
                - selected_index: Index of the selected patch
        """
        result = self.predictor.predict(patch_embeddings, fold_idx, return_attention=True)
        attention = result["attention"]
        if attention is None:
            raise ValueError("Model did not return attention weights")
        attention = attention.flatten()

        if patch_embeddings.dim() == 3:
            patch_embeddings = patch_embeddings.squeeze(0)

        threshold = torch.quantile(attention, threshold_quantile)
        mask = attention >= threshold
        selected_patches = patch_embeddings[mask].cpu()

        mean_emb = selected_patches.mean(dim=0)
        similarities = F.cosine_similarity(selected_patches, mean_emb.unsqueeze(0), dim=1)
        local_index = similarities.argmax().item()
        selected_index = mask.nonzero(as_tuple=False)[local_index].item()
        slide_embedding = patch_embeddings[selected_index].clone().cpu()

        if normalize:
            slide_embedding = slide_embedding / slide_embedding.norm()

        return {
            "slide_embedding": slide_embedding,
            "attention": attention,
            "probs": result["probs"],
            "pred_class": result["pred_class"],
            "selected_index": selected_index,
        }

    def compute_attention_filtered_nearest_euclidean(
        self,
        patch_embeddings: torch.Tensor,
        fold_idx: int,
        normalize: bool = False,
        threshold_quantile: float = 0.5,
    ) -> dict:
        """Compute slide embedding from attention-filtered patches, nearest to their mean (Euclidean).

        Selects patches with attention >= threshold_quantile quantile,
        computes their mean embedding, then returns the patch with the
        smallest Euclidean distance to that mean.

        Args:
            patch_embeddings: Tensor of shape (n_patches, embed_dim)
            fold_idx: Fold index to use for attention computation
            normalize: Whether to L2-normalize the output embedding
            threshold_quantile: Quantile threshold for attention filtering (default: 0.5 = median)

        Returns:
            dict with keys:
                - slide_embedding: Selected patch embedding (embed_dim,)
                - attention: Full attention weights (n_patches,)
                - probs: Prediction probabilities
                - pred_class: Predicted class
                - selected_index: Index of the selected patch
        """
        result = self.predictor.predict(patch_embeddings, fold_idx, return_attention=True)
        attention = result["attention"]
        if attention is None:
            raise ValueError("Model did not return attention weights")
        attention = attention.flatten()

        if patch_embeddings.dim() == 3:
            patch_embeddings = patch_embeddings.squeeze(0)

        threshold = torch.quantile(attention, threshold_quantile)
        mask = attention >= threshold
        selected_patches = patch_embeddings[mask].cpu()

        mean_emb = selected_patches.mean(dim=0)
        distances = torch.cdist(mean_emb.unsqueeze(0), selected_patches).squeeze(0)
        local_index = distances.argmin().item()
        selected_index = mask.nonzero(as_tuple=False)[local_index].item()
        slide_embedding = patch_embeddings[selected_index].clone().cpu()

        if normalize:
            slide_embedding = slide_embedding / slide_embedding.norm()

        return {
            "slide_embedding": slide_embedding,
            "attention": attention,
            "probs": result["probs"],
            "pred_class": result["pred_class"],
            "selected_index": selected_index,
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
                    result = self.compute_abmil(x, fold_idx, normalize=normalize)
                    embeddings.append(result["slide_embedding"].numpy())
                    labels.append(label)
        else:
            # Use fold 0 for all samples
            for idx in range(len(dataset)):
                x, label = dataset[idx]
                result = self.compute_abmil(x, fold_idx=0, normalize=normalize)
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
                    result = self.compute_abmil(x, fold_idx, normalize=normalize)

                    embeddings.append(result["slide_embedding"].numpy())
                    labels.append(label)
                    predictions.append(result["pred_class"])
                    probabilities.append(result["probs"].numpy())
                    indices.append(idx)
        else:
            for idx in range(len(dataset)):
                x, label = dataset[idx]
                result = self.compute_abmil(x, fold_idx=0, normalize=normalize)

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

    def save_to_hdf5(
        self,
        h5_path: str | Path,
        slide_embedding: np.ndarray,
        attention: np.ndarray | None = None,
        prediction: int | None = None,
        probabilities: np.ndarray | None = None,
        method_name: str | None = None,
        selected_index: int | None = None,
    ) -> None:
        """Save slide embedding to existing HDF5 file.

        Saves under 'slide_embedding/{method_name}/' group.

        Args:
            h5_path: Path to the HDF5 file
            slide_embedding: Slide-level embedding array
            attention: Attention weights (optional)
            prediction: Predicted class (optional)
            probabilities: Prediction probabilities (optional)
            method_name: Method name for group path. If None, uses self.method_name.
            selected_index: Index of the selected patch (optional)
        """
        if method_name is None:
            method_name = self.method_name

        group_path = f"slide_embedding/{method_name}"

        with h5py.File(h5_path, "a") as f:
            # Remove existing group if present
            if group_path in f:
                del f[group_path]

            grp = f.create_group(group_path)
            grp.create_dataset("embedding", data=slide_embedding)

            if attention is not None:
                grp.create_dataset("attention", data=attention)

            if prediction is not None:
                grp.attrs["prediction"] = prediction

            if probabilities is not None:
                grp.create_dataset("probabilities", data=probabilities)

            if selected_index is not None:
                grp.attrs["selected_index"] = selected_index

    def compute_and_save(
        self,
        dataset,
        use_val_fold: bool = True,
        normalize: bool = False,
        save_attention: bool = True,
        save_prediction: bool = True,
    ) -> dict:
        """Compute slide embeddings and save to HDF5 files.

        Each slide embedding is saved to its corresponding HDF5 file
        under 'slide_embedding/{method_name}/'.

        Args:
            dataset: Dataset with h5_files attribute and __getitem__ returning (x, label)
            use_val_fold: Use validation fold model for each sample
            normalize: Whether to L2-normalize embeddings
            save_attention: Whether to save attention weights
            save_prediction: Whether to save prediction results

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
                    result = self.compute_abmil(x, fold_idx, normalize=normalize)

                    slide_emb = result["slide_embedding"].numpy()
                    att = result["attention"].numpy() if result["attention"] is not None else None
                    pred = result["pred_class"]
                    probs = result["probs"].numpy()

                    # Save to HDF5
                    h5_path = dataset.h5_files[idx]
                    self.save_to_hdf5(
                        h5_path=h5_path,
                        slide_embedding=slide_emb,
                        attention=att if save_attention else None,
                        prediction=pred if save_prediction else None,
                        probabilities=probs if save_prediction else None,
                    )

                    embeddings.append(slide_emb)
                    labels.append(label)
                    predictions.append(pred)
                    probabilities.append(probs)
                    indices.append(idx)

                    print(f"Saved slide embedding to {h5_path}")
        else:
            for idx in range(len(dataset)):
                x, label = dataset[idx]
                result = self.compute_abmil(x, fold_idx=0, normalize=normalize)

                slide_emb = result["slide_embedding"].numpy()
                att = result["attention"].numpy() if result["attention"] is not None else None
                pred = result["pred_class"]
                probs = result["probs"].numpy()

                # Save to HDF5
                h5_path = dataset.h5_files[idx]
                self.save_to_hdf5(
                    h5_path=h5_path,
                    slide_embedding=slide_emb,
                    attention=att if save_attention else None,
                    prediction=pred if save_prediction else None,
                    probabilities=probs if save_prediction else None,
                )

                embeddings.append(slide_emb)
                labels.append(label)
                predictions.append(pred)
                probabilities.append(probs)
                indices.append(idx)

                print(f"Saved slide embedding to {h5_path}")

        return {
            "embeddings": np.array(embeddings),
            "labels": np.array(labels),
            "predictions": np.array(predictions),
            "probabilities": np.array(probabilities),
            "indices": indices,
        }

    _MODEL_FREE_STRATEGIES = {"nearest_cosine", "nearest_euclidean"}
    _MODEL_STRATEGIES = {
        "attention_top",
        "attention_nearest_cosine",
        "attention_nearest_euclidean",
        "attention_filtered_nearest_cosine",
        "attention_filtered_nearest_euclidean",
    }
    _ATTENTION_FILTERED_STRATEGIES = {
        "attention_filtered_nearest_cosine",
        "attention_filtered_nearest_euclidean",
    }
    _ALL_STRATEGIES = _MODEL_FREE_STRATEGIES | _MODEL_STRATEGIES

    def compute_and_save_strategy(
        self,
        dataset,
        strategy: str,
        use_val_fold: bool = True,
        normalize: bool = False,
        save_attention: bool = True,
        threshold_quantile: float = 0.5,
    ) -> dict:
        """Compute slide embeddings using a specified strategy and save to HDF5 files.

        Strategies:
            - "nearest_cosine": Patch nearest to mean by cosine similarity (no model needed)
            - "nearest_euclidean": Patch nearest to mean by Euclidean distance (no model needed)
            - "attention_top": Patch with highest attention (model needed)
            - "attention_nearest_cosine": Patch nearest to attention-weighted embedding by cosine
            - "attention_nearest_euclidean": Patch nearest to attention-weighted embedding by Euclidean
            - "attention_filtered_nearest_cosine": Patch nearest to mean of attention-filtered patches by cosine
            - "attention_filtered_nearest_euclidean": Patch nearest to mean of attention-filtered patches by Euclidean

        Args:
            dataset: Dataset with h5_files attribute and __getitem__ returning (x, label)
            strategy: Embedding strategy name
            use_val_fold: Use validation fold model (only for model-dependent strategies)
            normalize: Whether to L2-normalize embeddings
            save_attention: Whether to save attention weights (model-dependent strategies only)
            threshold_quantile: Quantile threshold for attention filtering (attention_filtered_* strategies only)

        Returns:
            dict with keys:
                - embeddings: np.ndarray of shape (n_samples, embed_dim)
                - labels: np.ndarray of shape (n_samples,)
                - indices: list of sample indices
                - selected_indices: list of selected patch indices
        """
        if strategy not in self._ALL_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Must be one of: {sorted(self._ALL_STRATEGIES)}"
            )

        # Determine HDF5 group name
        if strategy in self._MODEL_FREE_STRATEGIES:
            save_method_name = strategy
        elif strategy in self._ATTENTION_FILTERED_STRATEGIES:
            quantile_pct = int(threshold_quantile * 100)
            save_method_name = f"{self.method_name}_{strategy}_q{quantile_pct}"
        else:
            save_method_name = f"{self.method_name}_{strategy}"

        embeddings = []
        labels = []
        indices = []
        selected_indices = []

        if strategy in self._MODEL_FREE_STRATEGIES:
            compute_fn = (
                self.compute_nearest_cosine
                if strategy == "nearest_cosine"
                else self.compute_nearest_euclidean
            )

            for idx in range(len(dataset)):
                x, label = dataset[idx]
                result = compute_fn(x, normalize=normalize)

                slide_emb = result["slide_embedding"].numpy()
                h5_path = dataset.h5_files[idx]
                self.save_to_hdf5(
                    h5_path=h5_path,
                    slide_embedding=slide_emb,
                    method_name=save_method_name,
                    selected_index=result["selected_index"],
                )

                embeddings.append(slide_emb)
                labels.append(label)
                indices.append(idx)
                selected_indices.append(result["selected_index"])
                print(f"Saved slide embedding ({strategy}) to {h5_path}")

        else:
            # Model-dependent strategies
            if strategy == "attention_filtered_nearest_cosine":
                _tq = threshold_quantile
                compute_fn = lambda x, fold_idx, normalize=False: self.compute_attention_filtered_nearest_cosine(
                    x, fold_idx, normalize=normalize, threshold_quantile=_tq
                )
            elif strategy == "attention_filtered_nearest_euclidean":
                _tq = threshold_quantile
                compute_fn = lambda x, fold_idx, normalize=False: self.compute_attention_filtered_nearest_euclidean(
                    x, fold_idx, normalize=normalize, threshold_quantile=_tq
                )
            else:
                compute_fn_map = {
                    "attention_top": self.compute_top_attention,
                    "attention_nearest_cosine": self.compute_nearest_attention_cosine,
                    "attention_nearest_euclidean": self.compute_nearest_attention_euclidean,
                }
                compute_fn = compute_fn_map[strategy]

            if use_val_fold:
                for fold_idx in range(self.predictor.num_folds):
                    _, val_indices = self.predictor.get_fold_indices(fold_idx)

                    for idx in val_indices:
                        x, label = dataset[idx]
                        result = compute_fn(x, fold_idx, normalize=normalize)

                        slide_emb = result["slide_embedding"].numpy()
                        att = result["attention"].numpy() if result["attention"] is not None else None

                        h5_path = dataset.h5_files[idx]
                        self.save_to_hdf5(
                            h5_path=h5_path,
                            slide_embedding=slide_emb,
                            attention=att if save_attention else None,
                            method_name=save_method_name,
                            selected_index=result["selected_index"],
                        )

                        embeddings.append(slide_emb)
                        labels.append(label)
                        indices.append(idx)
                        selected_indices.append(result["selected_index"])
                        print(f"Saved slide embedding ({strategy}) to {h5_path}")
            else:
                for idx in range(len(dataset)):
                    x, label = dataset[idx]
                    result = compute_fn(x, fold_idx=0, normalize=normalize)

                    slide_emb = result["slide_embedding"].numpy()
                    att = result["attention"].numpy() if result["attention"] is not None else None

                    h5_path = dataset.h5_files[idx]
                    self.save_to_hdf5(
                        h5_path=h5_path,
                        slide_embedding=slide_emb,
                        attention=att if save_attention else None,
                        method_name=save_method_name,
                        selected_index=result["selected_index"],
                    )

                    embeddings.append(slide_emb)
                    labels.append(label)
                    indices.append(idx)
                    selected_indices.append(result["selected_index"])
                    print(f"Saved slide embedding ({strategy}) to {h5_path}")

        return {
            "embeddings": np.array(embeddings),
            "labels": np.array(labels),
            "indices": indices,
            "selected_indices": selected_indices,
        }

    @staticmethod
    def load_from_hdf5(
        h5_path: str | Path,
        method_name: str,
    ) -> dict:
        """Load slide embedding from HDF5 file.

        Args:
            h5_path: Path to the HDF5 file
            method_name: Method name used when saving

        Returns:
            dict with keys: embedding, attention (if exists), prediction, probabilities
        """
        group_path = f"slide_embedding/{method_name}"

        with h5py.File(h5_path, "r") as f:
            if group_path not in f:
                raise KeyError(f"Group '{group_path}' not found in {h5_path}")

            grp = f[group_path]
            result = {
                "embedding": grp["embedding"][:],
            }

            if "attention" in grp:
                result["attention"] = grp["attention"][:]

            if "prediction" in grp.attrs:
                result["prediction"] = grp.attrs["prediction"]

            if "probabilities" in grp:
                result["probabilities"] = grp["probabilities"][:]

            if "selected_index" in grp.attrs:
                result["selected_index"] = grp.attrs["selected_index"]

        return result

    @staticmethod
    def load_dataset_embeddings(
        data_dir: str | Path,
        method_name: str,
        csv_path: str | Path,
    ) -> dict:
        """Load slide embeddings for all samples from HDF5 files.

        Args:
            data_dir: Directory containing HDF5 files
            method_name: Method name used when saving (e.g., "abmil")
            csv_path: Path to CSV file with case_id and label columns

        Returns:
            dict with keys:
                - embeddings: np.ndarray of shape (n_samples, embed_dim)
                - labels: np.ndarray of shape (n_samples,)
                - predictions: np.ndarray or None
                - probabilities: np.ndarray or None
                - attentions: list of attention arrays or None
                - case_names: list of case IDs
                - h5_paths: list of Path objects
        """
        data_dir = Path(data_dir)
        df = pd.read_csv(csv_path)

        embeddings = []
        labels = []
        predictions = []
        probabilities = []
        attentions = []
        case_names = []
        h5_paths = []

        for _, row in df.iterrows():
            case_id = row["case_id"]
            label = row["label"]
            h5_path = data_dir / f"{case_id}.h5"

            if not h5_path.exists():
                print(f"Warning: {h5_path} not found, skipping.")
                continue

            try:
                data = SlideEmbeddingCalculator.load_from_hdf5(str(h5_path), method_name)
            except KeyError:
                print(f"Warning: No slide embedding for {case_id}, skipping.")
                continue

            embeddings.append(data["embedding"])
            labels.append(label)
            predictions.append(data.get("prediction"))
            probabilities.append(data.get("probabilities"))
            attentions.append(data.get("attention"))
            case_names.append(case_id)
            h5_paths.append(h5_path)

        has_predictions = predictions and predictions[0] is not None
        has_probabilities = probabilities and probabilities[0] is not None

        return {
            "embeddings": np.array(embeddings),
            "labels": np.array(labels),
            "predictions": np.array(predictions) if has_predictions else None,
            "probabilities": np.array(probabilities) if has_probabilities else None,
            "attentions": attentions,
            "case_names": case_names,
            "h5_paths": h5_paths,
        }
