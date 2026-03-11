"""Slide-level embedding computation from patch embeddings and attention."""

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from .predictor import MILPredictor

_MODEL_FREE_METHODS = {"nearest_cosine", "nearest_euclidean"}


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
        result = calculator.compute_abmil(patch_embeddings, fold_idx=0)

        # Dataset (saves to HDF5)
        results = calculator.compute_and_save(dataset, method="abmil")
        results = calculator.compute_and_save(dataset, method="abmil_top")
        results = calculator.compute_and_save(dataset, method="nearest_cosine")
    """

    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        output_dir: str | Path,
        version: int | str = "latest",
        device: str = "auto",
        mil_model_name: str | None = None,
        inference_output_dir: str | Path | None = None,
    ):
        """Initialize the calculator.

        Args:
            model_class: Model class (e.g., MILModel)
            model_kwargs: Keyword arguments for model instantiation
            output_dir: 訓練出力のベースディレクトリ
            version: 使用するバージョン。"latest" で最新、int で version_X を指定
            device: Device to use ("auto", "cuda", "cpu")
            mil_model_name: MILモデルのアーキテクチャ名 (e.g., "abmil").
                            If None, extracted from model_config.
            inference_output_dir: 推論結果のベースディレクトリ。指定時は trainの
                                  version_X 名と同期したサブディレクトリを作成する。
        """
        self.predictor = MILPredictor(
            model_class=model_class,
            model_kwargs=model_kwargs,
            output_dir=output_dir,
            version=version,
            device=device,
        )

        # Extract MIL model name for HDF5 group prefix
        if mil_model_name is not None:
            self.mil_model_name = mil_model_name
        elif "model_config" in model_kwargs:
            # Extract from model_config (e.g., "abmil.base.gigapath.none" -> "abmil")
            self.mil_model_name = model_kwargs["model_config"].split(".")[0]
        else:
            self.mil_model_name = "mil"

        # trainのversion_X名に同期した推論出力ディレクトリを作成
        if inference_output_dir is not None:
            train_version_name = self.predictor.output_dir.name  # e.g., "version_0"
            self.inference_version_dir = Path(inference_output_dir) / train_version_name
            self.inference_version_dir.mkdir(parents=True, exist_ok=True)
            print(f"Inference output directory: {self.inference_version_dir}")
        else:
            self.inference_version_dir = None

    def _save_inference_config(self, method: str, **kwargs) -> None:
        """inference_version_dir に config.yaml を保存する。"""
        if self.inference_version_dir is None:
            return
        config = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_version_dir": str(self.predictor.output_dir),
            "mil_model_name": self.mil_model_name,
            "method": method,
            **kwargs,
        }
        config_path = self.inference_version_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def load_models(self, checkpoint_name: str = "best") -> None:
        """Load models from all folds.

        Args:
            checkpoint_name: Checkpoint name to load ("best" or "last")
        """
        self.predictor.load_models(checkpoint_name)

    # ------------------------------------------------------------------
    # Single-sample compute methods
    # ------------------------------------------------------------------

    def compute_abmil(
        self,
        patch_embeddings: torch.Tensor,
        fold_idx: int,
        normalize: bool = False,
    ) -> dict:
        """Compute slide embedding as attention-weighted sum of patch embeddings.

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

    def compute_abmil_top(
        self,
        patch_embeddings: torch.Tensor,
        fold_idx: int,
        normalize: bool = False,
    ) -> dict:
        """Compute slide embedding using the patch with highest attention.

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
        result = self.predictor.predict(patch_embeddings, fold_idx, return_attention=True)
        attention = result["attention"]
        if attention is None:
            raise ValueError("Model did not return attention weights")
        attention = attention.flatten()

        if patch_embeddings.dim() == 3:
            patch_embeddings = patch_embeddings.squeeze(0)

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

    def compute_abmil_nearest_cosine(
        self,
        patch_embeddings: torch.Tensor,
        fold_idx: int,
        normalize: bool = False,
    ) -> dict:
        """Compute slide embedding: patch nearest to attention-weighted embedding (cosine).

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
        result = self.compute_abmil(patch_embeddings, fold_idx, normalize=False)
        attention_emb = result["slide_embedding"]
        attention = result["attention"]

        if patch_embeddings.dim() == 3:
            patch_embeddings = patch_embeddings.squeeze(0)

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

    def compute_abmil_nearest_euclidean(
        self,
        patch_embeddings: torch.Tensor,
        fold_idx: int,
        normalize: bool = False,
    ) -> dict:
        """Compute slide embedding: patch nearest to attention-weighted embedding (Euclidean).

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
        result = self.compute_abmil(patch_embeddings, fold_idx, normalize=False)
        attention_emb = result["slide_embedding"]
        attention = result["attention"]

        if patch_embeddings.dim() == 3:
            patch_embeddings = patch_embeddings.squeeze(0)

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

    def compute_abmil_filtered_nearest_cosine(
        self,
        patch_embeddings: torch.Tensor,
        fold_idx: int,
        normalize: bool = False,
        threshold_quantile: float = 0.5,
    ) -> dict:
        """Compute slide embedding from attention-filtered patches, nearest to mean (cosine).

        Args:
            patch_embeddings: Tensor of shape (n_patches, embed_dim)
            fold_idx: Fold index to use for attention computation
            normalize: Whether to L2-normalize the output embedding
            threshold_quantile: Quantile threshold for attention filtering (default: 0.5)

        Returns:
            dict with keys:
                - slide_embedding: Selected patch embedding (embed_dim,)
                - attention: Full attention weights (n_patches,)
                - probs: None
                - pred_class: None
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
            "probs": None,
            "pred_class": None,
            "selected_index": selected_index,
        }

    def compute_abmil_filtered_nearest_euclidean(
        self,
        patch_embeddings: torch.Tensor,
        fold_idx: int,
        normalize: bool = False,
        threshold_quantile: float = 0.5,
    ) -> dict:
        """Compute slide embedding from attention-filtered patches, nearest to mean (Euclidean).

        Args:
            patch_embeddings: Tensor of shape (n_patches, embed_dim)
            fold_idx: Fold index to use for attention computation
            normalize: Whether to L2-normalize the output embedding
            threshold_quantile: Quantile threshold for attention filtering (default: 0.5)

        Returns:
            dict with keys:
                - slide_embedding: Selected patch embedding (embed_dim,)
                - attention: Full attention weights (n_patches,)
                - probs: None
                - pred_class: None
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
            "probs": None,
            "pred_class": None,
            "selected_index": selected_index,
        }

    def compute_nearest_cosine(
        self,
        patch_embeddings: torch.Tensor,
        normalize: bool = False,
    ) -> dict:
        """Compute slide embedding: patch nearest to mean by cosine similarity (no model).

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
        """Compute slide embedding: patch nearest to mean by Euclidean distance (no model).

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
            aggregation: How to aggregate ("mean" or "vote")

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

    # ------------------------------------------------------------------
    # Dataset-level compute + save
    # ------------------------------------------------------------------

    def compute_and_save(
        self,
        dataset,
        method: str | None = None,
        use_val_fold: bool = True,
        normalize: bool = False,
        save_attention: bool = True,
        save_prediction: bool = True,
        threshold_quantile: float = 0.5,
    ) -> dict:
        """Compute slide embeddings for the entire dataset and save to HDF5 files.

        Available methods:
            - self.mil_model_name (e.g. "abmil"): Attention-weighted sum
            - f"{self.mil_model_name}_top": Patch with highest attention
            - f"{self.mil_model_name}_nearest_cosine": Patch nearest to attention embedding (cosine)
            - f"{self.mil_model_name}_nearest_euclidean": Patch nearest to attention embedding (Euclidean)
            - f"{self.mil_model_name}_filtered_nearest_cosine": Attention-filtered nearest (cosine)
            - f"{self.mil_model_name}_filtered_nearest_euclidean": Attention-filtered nearest (Euclidean)
            - "nearest_cosine": Patch nearest to mean by cosine (no model)
            - "nearest_euclidean": Patch nearest to mean by Euclidean (no model)

        For patch-selecting methods (*_top, *_nearest_*, nearest_*), selected_index is
        saved to HDF5 as grp.attrs["selected_index"] and included in the return value.
        This enables direct use with save_selected_patch_images() for patch preview.

        Args:
            dataset: Dataset with h5_files attribute and __getitem__ returning (x, label)
            method: Embedding method name (also used as HDF5 group name).
                    Defaults to self.mil_model_name (e.g., "abmil").
            use_val_fold: Use validation fold model for each sample (model-based methods only)
            normalize: Whether to L2-normalize embeddings
            save_attention: Whether to save attention weights
            save_prediction: Whether to save prediction results (for methods that produce them)
            threshold_quantile: Quantile threshold for filtered methods (default: 0.5)

        Returns:
            dict with keys:
                - embeddings: np.ndarray of shape (n_samples, embed_dim)
                - labels: np.ndarray of shape (n_samples,)
                - predictions: np.ndarray or None
                - probabilities: np.ndarray or None
                - selected_indices: list or None (non-None for patch-selecting methods)
                - attentions: list of attention arrays or None
                - indices: list of dataset indices
                - h5_paths: list of Path objects (for use with save_selected_patch_images)
                - case_names: list of case IDs (for use with save_selected_patch_images)
        """
        if method is None:
            method = self.mil_model_name

        self._save_inference_config(
            method=method,
            use_val_fold=use_val_fold,
            normalize=normalize,
            save_attention=save_attention,
            save_prediction=save_prediction,
            threshold_quantile=threshold_quantile,
        )

        compute_fn = self._resolve_compute_fn(method, threshold_quantile)
        is_model_free = method in _MODEL_FREE_METHODS

        embeddings = []
        labels = []
        predictions = []
        probabilities = []
        selected_indices = []
        attentions = []
        indices = []
        h5_paths = []
        case_names = []

        if is_model_free or not use_val_fold:
            fold_iter = [(None, list(range(len(dataset))))]
        else:
            fold_iter = [
                (fold_idx, self.predictor.get_fold_indices(fold_idx)[1])
                for fold_idx in range(self.predictor.num_folds)
            ]

        for fold_idx, sample_indices in fold_iter:
            for idx in sample_indices:
                x, label = dataset[idx]

                if is_model_free:
                    result = compute_fn(x, normalize=normalize)
                else:
                    result = compute_fn(x, fold_idx, normalize=normalize)

                slide_emb = result["slide_embedding"].numpy()
                att = result["attention"].numpy() if result["attention"] is not None else None
                pred = result.get("pred_class")
                probs = result["probs"].numpy() if result.get("probs") is not None else None
                sel_idx = result.get("selected_index")

                h5_path = dataset.h5_files[idx]
                self.save_to_hdf5(
                    h5_path=h5_path,
                    slide_embedding=slide_emb,
                    attention=att if save_attention else None,
                    prediction=pred if save_prediction else None,
                    probabilities=probs if save_prediction else None,
                    method_name=method,
                    selected_index=sel_idx,
                )

                embeddings.append(slide_emb)
                labels.append(label)
                predictions.append(pred)
                probabilities.append(probs)
                selected_indices.append(sel_idx)
                attentions.append(att)
                indices.append(idx)
                h5_paths.append(h5_path)
                case_names.append(h5_path.stem)

                print(f"Saved slide embedding ({method}) to {h5_path}")

        has_predictions = any(p is not None for p in predictions)
        has_probabilities = any(p is not None for p in probabilities)
        has_selected = any(s is not None for s in selected_indices)
        has_attentions = any(a is not None for a in attentions)

        return {
            "embeddings": np.array(embeddings),
            "labels": np.array(labels),
            "predictions": np.array(predictions) if has_predictions else None,
            "probabilities": np.array(probabilities) if has_probabilities else None,
            "selected_indices": selected_indices if has_selected else None,
            "attentions": attentions if has_attentions else None,
            "indices": indices,
            "h5_paths": h5_paths,
            "case_names": case_names,
        }

    def _resolve_compute_fn(self, method: str, threshold_quantile: float = 0.5):
        """Resolve compute function from method name."""
        mn = self.mil_model_name
        filtered_cosine = f"{mn}_filtered_nearest_cosine"
        filtered_euclidean = f"{mn}_filtered_nearest_euclidean"

        if method == filtered_cosine or method.startswith(filtered_cosine):
            _tq = threshold_quantile
            return lambda x, fold_idx, normalize=False: self.compute_abmil_filtered_nearest_cosine(
                x, fold_idx, normalize=normalize, threshold_quantile=_tq
            )
        if method == filtered_euclidean or method.startswith(filtered_euclidean):
            _tq = threshold_quantile
            return lambda x, fold_idx, normalize=False: self.compute_abmil_filtered_nearest_euclidean(
                x, fold_idx, normalize=normalize, threshold_quantile=_tq
            )

        method_map = {
            mn: self.compute_abmil,
            f"{mn}_top": self.compute_abmil_top,
            f"{mn}_nearest_cosine": self.compute_abmil_nearest_cosine,
            f"{mn}_nearest_euclidean": self.compute_abmil_nearest_euclidean,
            "nearest_cosine": self.compute_nearest_cosine,
            "nearest_euclidean": self.compute_nearest_euclidean,
        }

        if method not in method_map:
            raise ValueError(
                f"Unknown method '{method}'. Available: {sorted(method_map) + [filtered_cosine, filtered_euclidean]}"
            )
        return method_map[method]

    # ------------------------------------------------------------------
    # HDF5 I/O
    # ------------------------------------------------------------------

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
            method_name: Method name for group path. If None, uses self.mil_model_name.
            selected_index: Index of the selected patch (optional)
        """
        if method_name is None:
            method_name = self.mil_model_name

        group_path = f"slide_embedding/{method_name}"

        with h5py.File(h5_path, "a") as f:
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

    @staticmethod
    def load_from_hdf5(
        h5_path: str | Path,
        method_name: str,
    ) -> dict:
        """Load slide embedding from HDF5 file.

        Args:
            h5_path: Path to the HDF5 file
            method_name: Method name used when saving (e.g., "abmil", "abmil_top")

        Returns:
            dict with keys: embedding, attention (if exists), prediction, probabilities,
                            selected_index (if exists)
        """
        group_path = f"slide_embedding/{method_name}"

        with h5py.File(h5_path, "r") as f:
            if group_path not in f:
                raise KeyError(f"Group '{group_path}' not found in {h5_path}")

            grp = f[group_path]
            result = {"embedding": grp["embedding"][:]}

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
            method_name: Method name used when saving (e.g., "abmil", "abmil_top")
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
