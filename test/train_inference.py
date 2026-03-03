"""End-to-end test: training + inference with all slide embedding methods."""

from pathlib import Path

from mil_toolbox.data import WSIDataset
from mil_toolbox.models import MILModel
from mil_toolbox.train import CrossValidationTrainer
from mil_toolbox.inference import SlideEmbeddingCalculator
from mil_toolbox.utils import (
    compute_metrics_from_results,
    compute_confusion_matrix,
    print_metrics,
    plot_umap,
    plot_confusion_matrix,
)


def main():
    data_dir = "test/data/embedding"
    csv_path = "test/data/labels.csv"
    output_dir = "./outputs"
    image_dir = Path("test/data/image")
    image_dir.mkdir(parents=True, exist_ok=True)

    encoder_name = "uni"
    model_name = "abmil"
    model_config = f"{model_name}.base.{encoder_name}.none"
    model_kwargs = {"num_classes": 2, "model_config": model_config}

    # ==============================
    # Training
    # ==============================
    print("=" * 50)
    print("Training")
    print("=" * 50)

    dataset = WSIDataset(data_dir, encoder_name, csv_path)
    print(f"Dataset: {len(dataset)} samples")

    cv_trainer = CrossValidationTrainer(
        model_class=MILModel,
        model_kwargs=model_kwargs,
        dataset=dataset,
        num_fold=5,
        output_dir=output_dir,
        max_epochs=2,
    )
    results_train = cv_trainer.run()

    # config.yaml と version_X ディレクトリの確認
    version_dir = cv_trainer.fold_manager.output_dir
    config_path = version_dir / "config.yaml"
    assert config_path.exists(), f"config.yaml not found: {config_path}"
    print(f"config.yaml: {config_path}")
    print(f"Best val losses: {[r['best_val_loss'] for r in results_train]}")

    # ==============================
    # Inference (version="latest" がデフォルト)
    # ==============================

    calculator = SlideEmbeddingCalculator(
        model_class=MILModel,
        model_kwargs=model_kwargs,
        output_dir=output_dir,
        device="cpu",
    )
    calculator.load_models(checkpoint_name="best")

    # version=0 で明示指定できることも確認
    calculator_v0 = SlideEmbeddingCalculator(
        model_class=MILModel,
        model_kwargs=model_kwargs,
        output_dir=output_dir,
        version=0,
        device="cpu",
    )
    calculator_v0.load_models(checkpoint_name="best")
    print(f"version=0 loaded {calculator_v0.predictor.num_folds} folds")

    class_names = {0: "Class0", 1: "Class1"}

    # ==============================
    # abmil (attention-weighted sum)
    # ==============================
    print("\n" + "=" * 50)
    print("Inference: abmil (attention-weighted sum)")
    print("=" * 50)

    results_abmil = calculator.compute_and_save(
        dataset, method="abmil", use_val_fold=True, save_attention=True, save_prediction=True
    )
    print(f"Embeddings shape: {results_abmil['embeddings'].shape}")

    metrics = compute_metrics_from_results(
        results_abmil, positive_class=1, class_names=class_names
    )
    print_metrics(metrics, title="ABMIL Metrics")

    cm = compute_confusion_matrix(results_abmil["labels"], results_abmil["predictions"])
    plot_confusion_matrix(
        cm,
        output_path=image_dir / "confusion_matrix_abmil.jpeg",
        class_names=class_names,
        title="ABMIL - Confusion Matrix",
    )
    plot_confusion_matrix(
        cm,
        output_path=image_dir / "confusion_matrix_abmil_normalized.jpeg",
        class_names=class_names,
        normalize=True,
        title="ABMIL - Confusion Matrix (Normalized)",
    )

    # ==============================
    # nearest_cosine
    # ==============================
    print("\n" + "=" * 50)
    print("Inference: nearest_cosine")
    print("=" * 50)

    results_cos = calculator.compute_and_save(dataset, method="nearest_cosine")
    print(f"Embeddings shape: {results_cos['embeddings'].shape}")
    print(f"Selected indices (first 5): {results_cos['selected_indices'][:5]}")

    # ==============================
    # nearest_euclidean
    # ==============================
    print("\n" + "=" * 50)
    print("Inference: nearest_euclidean")
    print("=" * 50)

    results_euc = calculator.compute_and_save(dataset, method="nearest_euclidean")
    print(f"Embeddings shape: {results_euc['embeddings'].shape}")
    print(f"Selected indices (first 5): {results_euc['selected_indices'][:5]}")

    # ==============================
    # abmil_top
    # ==============================
    print("\n" + "=" * 50)
    print("Inference: abmil_top")
    print("=" * 50)

    results_top = calculator.compute_and_save(dataset, method="abmil_top", use_val_fold=True)
    print(f"Embeddings shape: {results_top['embeddings'].shape}")
    print(f"Selected indices (first 5): {results_top['selected_indices'][:5]}")

    # ==============================
    # abmil_nearest_cosine
    # ==============================
    print("\n" + "=" * 50)
    print("Inference: abmil_nearest_cosine")
    print("=" * 50)

    results_att_cos = calculator.compute_and_save(
        dataset, method="abmil_nearest_cosine", use_val_fold=True
    )
    print(f"Embeddings shape: {results_att_cos['embeddings'].shape}")
    print(f"Selected indices (first 5): {results_att_cos['selected_indices'][:5]}")

    # ==============================
    # abmil_nearest_euclidean
    # ==============================
    print("\n" + "=" * 50)
    print("Inference: abmil_nearest_euclidean")
    print("=" * 50)

    results_att_euc = calculator.compute_and_save(
        dataset, method="abmil_nearest_euclidean", use_val_fold=True
    )
    print(f"Embeddings shape: {results_att_euc['embeddings'].shape}")
    print(f"Selected indices (first 5): {results_att_euc['selected_indices'][:5]}")

    # ==============================
    # UMAP for each method
    # ==============================
    methods = {
        "abmil": results_abmil,
        "nearest_cosine": results_cos,
        "nearest_euclidean": results_euc,
        "abmil_top": results_top,
        "abmil_nearest_cosine": results_att_cos,
        "abmil_nearest_euclidean": results_att_euc,
    }

    for method_name, results in methods.items():
        umap_data = {
            "embeddings": results["embeddings"],
            "labels": results["labels"],
            "predictions": results.get("predictions"),
        }
        plot_umap(
            umap_data,
            output_path=image_dir / f"umap_{method_name}.jpeg",
            class_names=class_names,
            title=f"UMAP - {method_name}",
        )

    # ==============================
    # Verify HDF5 loading
    # ==============================
    print("\n" + "=" * 50)
    print("Verify HDF5 loading")
    print("=" * 50)

    h5_path = dataset.h5_files[0]
    hdf5_names = [
        "abmil",
        "nearest_cosine",
        "nearest_euclidean",
        "abmil_top",
        "abmil_nearest_cosine",
        "abmil_nearest_euclidean",
    ]

    print("\n--- First HDF5 file ---")
    for name in hdf5_names:
        data = SlideEmbeddingCalculator.load_from_hdf5(str(h5_path), name)
        parts = [f"embedding={data['embedding'].shape}"]
        if "attention" in data:
            parts.append(f"attention={data['attention'].shape}")
        if "prediction" in data:
            parts.append(f"prediction={data['prediction']}")
        if "selected_index" in data:
            parts.append(f"selected_index={data['selected_index']}")
        print(f"  {name}: {', '.join(parts)}")

    print("\n--- All HDF5 files ---")
    for name in hdf5_names:
        loaded = SlideEmbeddingCalculator.load_dataset_embeddings(
            data_dir, name, csv_path
        )
        print(f"  load_dataset_embeddings({name}): {loaded['embeddings'].shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
