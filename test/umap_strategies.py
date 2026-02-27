"""Test UMAP visualization for all slide embedding strategies."""

from pathlib import Path

from mil_toolbox.data import WSIDataset
from mil_toolbox.models import MILModel
from mil_toolbox.inference import SlideEmbeddingCalculator
from mil_toolbox.utils import plot_umap


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
    # ABMIL: labels & predictions baseline
    # ==============================
    print("\n" + "=" * 50)
    print("Computing ABMIL embeddings (baseline for labels/predictions)")
    print("=" * 50)

    results_abmil = calculator.compute_and_save(
        dataset, use_val_fold=True, save_attention=True, save_prediction=True
    )
    base_labels = results_abmil["labels"]
    base_predictions = results_abmil["predictions"]

    # ==============================
    # Model-free strategies
    # ==============================
    results_cos = calculator.compute_and_save_strategy(
        dataset, strategy="nearest_cosine"
    )
    results_euc = calculator.compute_and_save_strategy(
        dataset, strategy="nearest_euclidean"
    )

    # ==============================
    # Model-dependent strategies
    # ==============================
    results_top = calculator.compute_and_save_strategy(
        dataset, strategy="attention_top", use_val_fold=True
    )
    results_att_cos = calculator.compute_and_save_strategy(
        dataset, strategy="attention_nearest_cosine", use_val_fold=True
    )
    results_att_euc = calculator.compute_and_save_strategy(
        dataset, strategy="attention_nearest_euclidean", use_val_fold=True
    )

    # ==============================
    # attention_filtered strategies (複数閾値)
    # ==============================
    threshold_quantiles = [0.25, 0.5, 0.75]
    results_filtered = {}

    for tq in threshold_quantiles:
        pct = int(tq * 100)
        for metric in ("cosine", "euclidean"):
            strategy = f"attention_filtered_nearest_{metric}"
            key = f"{strategy}_q{pct}"
            results_filtered[key] = calculator.compute_and_save_strategy(
                dataset,
                strategy=strategy,
                use_val_fold=True,
                threshold_quantile=tq,
            )

    # ==============================
    # UMAP plots
    # ==============================
    print("\n" + "=" * 50)
    print("Plotting UMAPs")
    print("=" * 50)

    base_strategies = {
        "abmil": results_abmil,
        "nearest_cosine": results_cos,
        "nearest_euclidean": results_euc,
        "attention_top": results_top,
        "attention_nearest_cosine": results_att_cos,
        "attention_nearest_euclidean": results_att_euc,
    }

    for strategy_name, results in base_strategies.items():
        plot_umap(
            {
                "embeddings": results["embeddings"],
                "labels": base_labels,
                "predictions": base_predictions,
            },
            output_path=image_dir / f"umap_{strategy_name}.jpeg",
            class_names=class_names,
            title=f"UMAP - {strategy_name}",
        )

    for key, results in results_filtered.items():
        plot_umap(
            {
                "embeddings": results["embeddings"],
                "labels": base_labels,
                "predictions": base_predictions,
            },
            output_path=image_dir / f"umap_{key}.jpeg",
            class_names=class_names,
            title=f"UMAP - {key}",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
