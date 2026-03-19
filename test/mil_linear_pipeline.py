"""MIL toolbox フルパイプラインテスト。

1. ダミー特徴量作成 (DummyWSIDataset)
2. ABMIL CV 訓練 (outputs/train)
3. ABMIL 推論 → confusion matrix + UMAP 保存 + スライド特徴量形状出力
4. DummyTitanModel 推論 → UMAP 保存
5. LinearProbe CV 訓練 (Titan 特徴量 + ABMIL と同じ fold)
6. LinearProbe 推論 → confusion matrix 保存

実行方法:
    python test/mil_linear_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import lightning as L

from mil_toolbox.data import DummyWSIDataset, EmbeddingDataset
from mil_toolbox.models import MILModel, LinearProbeModel
from mil_toolbox.train import CrossValidationTrainer
from mil_toolbox.inference import MILPredictor, SlideEmbeddingCalculator
from mil_toolbox.utils import (
    compute_metrics_from_results,
    compute_confusion_matrix,
    print_metrics,
    plot_umap,
    plot_confusion_matrix,
)

# ------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------
TRAIN_OUTPUT_DIR = "./outputs/train"
INFER_OUTPUT_DIR = "./outputs/inference"
NUM_WSI = 20
NUM_FOLD = 4
MAX_EPOCHS = 2
NUM_CLASSES = 2
CLASS_NAMES = {0: "Class0", 1: "Class1"}
PATCH_DIM = 1024
TITAN_OUT_DIM = 768


# ------------------------------------------------------------------
# DummyTitanModel: パッチ埋め込み → スライドレベル特徴量
# ------------------------------------------------------------------
class DummyTitanModel(L.LightningModule):
    """ダミーTitanスライドエンコーダー。

    実際のTitanはNvidiaのFoundation Modelだが、
    ここではランダム重みで動作するテスト用実装。
    パッチ埋め込みをmean poolingして線形変換し、スライドレベル特徴量を返す。

    Args:
        in_dim: 入力パッチ埋め込みの次元数。
        out_dim: 出力スライド特徴量の次元数。
    """

    def __init__(self, in_dim: int = PATCH_DIM, out_dim: int = TITAN_OUT_DIM):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    @torch.no_grad()
    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """パッチ埋め込みをスライドレベル特徴量に変換。

        Args:
            patches: 形状 (n_patches, in_dim) のパッチ埋め込み。

        Returns:
            slide_embedding: 形状 (out_dim,) のスライドレベル特徴量。
        """
        mean_pool = patches.mean(dim=0)       # (in_dim,)
        return self.encoder(mean_pool)        # (out_dim,)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ------------------------------------------------------------------
# ヘルパー: ABMIL 推論ループ
# DummyWSIDataset は h5_files を持たないため compute_and_save は使用不可
# ------------------------------------------------------------------
def infer_abmil(calculator: SlideEmbeddingCalculator, dataset) -> dict:
    """ABMIL の val fold 推論を手動ループで実行。

    Returns:
        dict:
            embeddings (N, D), labels (N,), predictions (N,)
            ※ 元のデータセット順にソート済み
    """
    predictor = calculator.predictor
    all_embs, all_labels, all_preds, all_indices = [], [], [], []

    for fold_idx in range(predictor.num_folds):
        _, val_indices = predictor.get_fold_indices(fold_idx)
        for idx in val_indices:
            patches, label = dataset[idx]
            result = calculator.compute_abmil(patches, fold_idx)
            all_embs.append(result["slide_embedding"].numpy())
            all_labels.append(label)
            all_preds.append(result["pred_class"])
            all_indices.append(idx)

    order = np.argsort(all_indices)
    return {
        "embeddings": np.array(all_embs)[order],
        "labels": np.array(all_labels)[order],
        "predictions": np.array(all_preds)[order],
    }


# ------------------------------------------------------------------
# ヘルパー: LinearProbe 推論ループ
# return_attention=False 必須 (LinearProbeModel は attention=None を返すため)
# ------------------------------------------------------------------
def infer_linear_probe(predictor: MILPredictor, embed_dataset: EmbeddingDataset) -> dict:
    """LinearProbeModel の val fold 推論を手動ループで実行。

    Returns:
        dict: labels (N,), predictions (N,) — 元のデータセット順にソート済み
    """
    all_preds, all_labels, all_indices = [], [], []

    for fold_idx in range(predictor.num_folds):
        _, val_indices = predictor.get_fold_indices(fold_idx)
        for idx in val_indices:
            x, label = embed_dataset[idx]   # (D,), int
            # attention=None のモデルなので return_attention=False 必須
            result = predictor.predict(x, fold_idx, return_attention=False)
            all_preds.append(result["pred_class"])
            all_labels.append(label)
            all_indices.append(idx)

    order = np.argsort(all_indices)
    return {
        "labels": np.array(all_labels)[order],
        "predictions": np.array(all_preds)[order],
    }


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================
    # Step 1: ダミー特徴量作成
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 1: Create DummyWSIDataset")
    print("=" * 60)

    dataset = DummyWSIDataset(num_wsi=NUM_WSI)
    print(f"Dataset: {len(dataset)} WSIs, patch_dim={PATCH_DIM}")

    # ============================================================
    # Step 2: ABMIL CV 訓練 (outputs/train)
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: ABMIL CrossValidation Training")
    print("=" * 60)

    model_config = "abmil.base.uni.none"
    abmil_kwargs = {"num_classes": NUM_CLASSES, "model_config": model_config}

    cv_trainer = CrossValidationTrainer(
        model_class=MILModel,
        model_kwargs=abmil_kwargs,
        dataset=dataset,
        num_fold=NUM_FOLD,
        output_dir=TRAIN_OUTPUT_DIR,
        max_epochs=MAX_EPOCHS,
        batch_size=2,
    )
    cv_trainer.run()

    train_version_dir: Path = cv_trainer.fold_manager.output_dir
    train_version_idx = int(train_version_dir.name.split("_")[1])
    print(f"Train version dir: {train_version_dir}")

    # 推論結果の保存先 (outputs/inference/version_X — 訓練と同一バージョン番号)
    infer_version_dir = Path(INFER_OUTPUT_DIR) / train_version_dir.name
    infer_version_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Step 3: ABMIL 推論
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3: ABMIL Inference")
    print("=" * 60)

    calculator = SlideEmbeddingCalculator(
        model_class=MILModel,
        model_kwargs=abmil_kwargs,
        output_dir=TRAIN_OUTPUT_DIR,      # モデルは train dir からロード
        version=train_version_idx,
        device="cpu",
    )
    calculator.load_models()

    results_abmil = infer_abmil(calculator, dataset)

    # スライドレベル特徴量の形状出力
    print(f"ABMIL slide embeddings shape: {results_abmil['embeddings'].shape}")

    # Confusion Matrix 保存
    cm_abmil = compute_confusion_matrix(
        results_abmil["labels"], results_abmil["predictions"]
    )
    plot_confusion_matrix(
        cm_abmil,
        output_path=infer_version_dir / "confusion_matrix_abmil.jpeg",
        class_names=CLASS_NAMES,
        title="ABMIL - Confusion Matrix",
    )
    print(f"Saved: {infer_version_dir / 'confusion_matrix_abmil.jpeg'}")

    # UMAP 保存
    plot_umap(
        data={
            "embeddings": results_abmil["embeddings"],
            "labels": results_abmil["labels"],
            "predictions": results_abmil["predictions"],
        },
        output_path=infer_version_dir / "umap_abmil.jpeg",
        class_names=CLASS_NAMES,
        title="ABMIL - UMAP",
        annotate=False,
    )
    print(f"Saved: {infer_version_dir / 'umap_abmil.jpeg'}")

    metrics_abmil = compute_metrics_from_results(
        results_abmil, positive_class=1, class_names=CLASS_NAMES
    )
    print_metrics(metrics_abmil, title="ABMIL Metrics")

    # ============================================================
    # Step 4: DummyTitanModel 推論
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 4: DummyTitan Inference")
    print("=" * 60)

    titan_model = DummyTitanModel(in_dim=PATCH_DIM, out_dim=TITAN_OUT_DIM)
    titan_model.eval()

    titan_embeddings: list[np.ndarray] = []
    titan_labels: list[int] = []
    for idx in range(len(dataset)):
        patches, label = dataset[idx]
        emb = titan_model.encode(patches)   # (TITAN_OUT_DIM,)
        titan_embeddings.append(emb.numpy())
        titan_labels.append(label)

    titan_emb_arr = np.array(titan_embeddings)   # (N, TITAN_OUT_DIM)
    titan_lbl_arr = np.array(titan_labels)        # (N,)
    print(f"Titan slide embeddings shape: {titan_emb_arr.shape}")

    # ============================================================
    # Step 5: LinearProbe CV 訓練 (Titan 特徴量 + ABMIL fold 再利用)
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 5: LinearProbe CrossValidation Training (Titan embeddings)")
    print("=" * 60)

    embed_dataset = EmbeddingDataset(titan_emb_arr, titan_lbl_arr.tolist())
    print(
        f"EmbeddingDataset: {len(embed_dataset)} samples, "
        f"embedding_dim={embed_dataset.embedding_dim}"
    )

    lp_trainer = CrossValidationTrainer(
        model_class=LinearProbeModel,
        model_kwargs={"embedding_dim": TITAN_OUT_DIM, "num_classes": NUM_CLASSES},
        dataset=embed_dataset,
        num_fold=NUM_FOLD,
        output_dir=TRAIN_OUTPUT_DIR,
        max_epochs=MAX_EPOCHS,
        batch_size=4,
        collate_fn=None,                          # デフォルトコレート (EmbeddingDataset 対応)
        existing_fold_dir=train_version_dir,      # ABMIL と同じ fold 分割を使用
    )
    lp_trainer.run()

    lp_version_dir: Path = lp_trainer.fold_manager.output_dir
    lp_version_idx = int(lp_version_dir.name.split("_")[1])
    print(f"LinearProbe version dir: {lp_version_dir}")

    # LinearProbe 推論の保存先は LinearProbe の train version に合わせる
    lp_infer_version_dir = Path(INFER_OUTPUT_DIR) / lp_version_dir.name
    lp_infer_version_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Step 6: LinearProbe 推論 → Confusion Matrix + UMAP 保存
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 6: LinearProbe Inference")
    print("=" * 60)

    lp_predictor = MILPredictor(
        model_class=LinearProbeModel,
        model_kwargs={"embedding_dim": TITAN_OUT_DIM, "num_classes": NUM_CLASSES},
        output_dir=TRAIN_OUTPUT_DIR,
        version=lp_version_idx,
        device="cpu",
    )
    lp_predictor.load_models()

    results_lp = infer_linear_probe(lp_predictor, embed_dataset)

    cm_lp = compute_confusion_matrix(results_lp["labels"], results_lp["predictions"])
    plot_confusion_matrix(
        cm_lp,
        output_path=lp_infer_version_dir / "confusion_matrix_linear_probe.jpeg",
        class_names=CLASS_NAMES,
        title="LinearProbe - Confusion Matrix",
    )
    print(f"Saved: {lp_infer_version_dir / 'confusion_matrix_linear_probe.jpeg'}")

    metrics_lp = compute_metrics_from_results(
        results_lp, positive_class=1, class_names=CLASS_NAMES
    )
    print_metrics(metrics_lp, title="LinearProbe Metrics")

    # UMAP 保存 (Titan 特徴量空間 + LinearProbe 予測による誤分類マーク)
    plot_umap(
        data={
            "embeddings": titan_emb_arr,
            "labels": titan_lbl_arr,
            "predictions": results_lp["predictions"],
        },
        output_path=lp_infer_version_dir / "umap_titan_linear_probe.jpeg",
        class_names=CLASS_NAMES,
        title="DummyTitan (LinearProbe predictions) - UMAP",
        annotate=False,
        show_misclassified=True,
    )
    print(f"Saved: {lp_infer_version_dir / 'umap_titan_linear_probe.jpeg'}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print(f"  ABMIL train:            {train_version_dir}")
    print(f"  LinearProbe train:      {lp_version_dir}")
    print(f"  ABMIL inference:        {infer_version_dir}")
    print(f"  LinearProbe inference:  {lp_infer_version_dir}")
    print(f"  ABMIL inference files:")
    for f in sorted(infer_version_dir.glob("*")):
        print(f"    {f.name}")
    print(f"  LinearProbe inference files:")
    for f in sorted(lp_infer_version_dir.glob("*")):
        print(f"    {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
