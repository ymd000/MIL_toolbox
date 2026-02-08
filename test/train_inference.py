"""End-to-end test: inference with all slide embedding strategies using dummy data."""

from pathlib import Path

from mil_toolbox.data import WSIDataset
from mil_toolbox.models import MILModel
from mil_toolbox.inference import SlideEmbeddingCalculator
from mil_toolbox.utils import compute_metrics_from_results, print_metrics, plot_umap


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

    dataset = WSIDataset(data_dir, encoder_name, csv_path)
    print(f"Dataset: {len(dataset)} samples")

    calculator = SlideEmbeddingCalculator(
        model_class=MILModel,
        model_kwargs=model_kwargs,
        output_dir=output_dir,
        device="cpu",
    )
    calculator.load_models(checkpoint_name="best")

    class_names = {0: "Class0", 1: "Class1"}

    # ==============================
    # ABMIL (attention-weighted sum)
    # ==============================
    print("\n" + "=" * 50)
    print("Inference: ABMIL (attention-weighted sum)")
    print("=" * 50)

    results_abmil = calculator.compute_and_save(
        dataset, use_val_fold=True, save_attention=True, save_prediction=True
    )
    print(f"Embeddings shape: {results_abmil['embeddings'].shape}")

    metrics = compute_metrics_from_results(
        results_abmil, positive_class=1, class_names=class_names
    )
    print_metrics(metrics, title="ABMIL Metrics")

    # Store ABMIL predictions for UMAP
    abmil_predictions = results_abmil["predictions"]
    abmil_labels = results_abmil["labels"]

    # ==============================
    # nearest_cosine
    # ==============================
    print("\n" + "=" * 50)
    print("Inference: nearest_cosine")
    print("=" * 50)

    results_cos = calculator.compute_and_save_strategy(
        dataset, strategy="nearest_cosine"
    )
    print(f"Embeddings shape: {results_cos['embeddings'].shape}")
    print(f"Selected indices (first 5): {results_cos['selected_indices'][:5]}")

    # ==============================
    # nearest_euclidean
    # ==============================
    print("\n" + "=" * 50)
    print("Inference: nearest_euclidean")
    print("=" * 50)

    results_euc = calculator.compute_and_save_strategy(
        dataset, strategy="nearest_euclidean"
    )
    print(f"Embeddings shape: {results_euc['embeddings'].shape}")
    print(f"Selected indices (first 5): {results_euc['selected_indices'][:5]}")

    # ==============================
    # top_attention
    # ==============================
    print("\n" + "=" * 50)
    print("Inference: top_attention")
    print("=" * 50)

    results_top = calculator.compute_and_save_strategy(
        dataset, strategy="top_attention", use_val_fold=True
    )
    print(f"Embeddings shape: {results_top['embeddings'].shape}")
    print(f"Selected indices (first 5): {results_top['selected_indices'][:5]}")

    # ==============================
    # nearest_attention_cosine
    # ==============================
    print("\n" + "=" * 50)
    print("Inference: nearest_attention_cosine")
    print("=" * 50)

    results_att_cos = calculator.compute_and_save_strategy(
        dataset, strategy="nearest_attention_cosine", use_val_fold=True
    )
    print(f"Embeddings shape: {results_att_cos['embeddings'].shape}")
    print(f"Selected indices (first 5): {results_att_cos['selected_indices'][:5]}")

    # ==============================
    # nearest_attention_euclidean
    # ==============================
    print("\n" + "=" * 50)
    print("Inference: nearest_attention_euclidean")
    print("=" * 50)

    results_att_euc = calculator.compute_and_save_strategy(
        dataset, strategy="nearest_attention_euclidean", use_val_fold=True
    )
    print(f"Embeddings shape: {results_att_euc['embeddings'].shape}")
    print(f"Selected indices (first 5): {results_att_euc['selected_indices'][:5]}")

    # ==============================
    # UMAP for each strategy
    # ==============================
    strategies = {
        "abmil": results_abmil,
        "nearest_cosine": results_cos,
        "nearest_euclidean": results_euc,
        "top_attention": results_top,
        "nearest_attention_cosine": results_att_cos,
        "nearest_attention_euclidean": results_att_euc,
    }

    for strategy_name, results in strategies.items():
        umap_data = {
            "embeddings": results["embeddings"],
            "labels": abmil_labels,
            "predictions": abmil_predictions,
        }
        plot_umap(
            umap_data,
            output_path=image_dir / f"umap_{strategy_name}.jpeg",
            class_names=class_names,
            title=f"UMAP - {strategy_name}",
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
        "abmil_top_attention",
        "abmil_nearest_attention_cosine",
        "abmil_nearest_attention_euclidean",
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
