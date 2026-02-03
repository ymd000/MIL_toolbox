from mil_toolbox.data import WSIDataset
from mil_toolbox.models import MILModel
from mil_toolbox.inference import SlideEmbeddingCalculator, PreviewAttention

import os
import numpy as np
import umap
import matplotlib.pyplot as plt


def main():
    root_dir = "/home/ymd/github.com/ymd000/SFT_Meni"

    image_dir = "data/image"
    image_dir_full = os.path.join(root_dir, image_dir)
    os.makedirs(image_dir_full, exist_ok=True)

    data_dir = "data/embedding"
    data_dir_full = os.path.join(root_dir, data_dir)
    csv_path = "data/label/case_labels_n69.csv"
    csv_path_full = os.path.join(root_dir, csv_path)

    encoder_name = "gigapath"

    dataset = WSIDataset(data_dir_full, encoder_name, csv_path_full)

    # for i in range(len(dataset)):
    #     wsi, label = dataset[i]
    #     print(f"WSI {i}: {wsi.shape}, Label {i}: {label}")

    model_name = "abmil"
    model_config = f"{model_name}.base.{encoder_name}.none"

    model_kwargs = {"num_classes": 2, "model_config": model_config}

    # num_fold = 5

    # print("\n" + "=" * 50)
    # print("Train with CrossValidationTrainer")
    # print("=" * 50)
    #
    # trainer = CrossValidationTrainer(
    #     model_class=MILModel,
    #     model_kwargs=model_kwargs,
    #     dataset=dataset,
    #     num_fold=num_fold,
    #     output_dir="./outputs"
    # )
    # trainer.run()

    print("\n" + "=" * 50)
    print("Inference with SlideEmbeddingCalculator")
    print("=" * 50)

    calculator = SlideEmbeddingCalculator(
        model_class=MILModel,
        model_kwargs=model_kwargs,
        output_dir="./outputs",
        device="cpu",
    )
    calculator.load_models(checkpoint_name="best")

    # Compute slide embeddings for all samples (using validation fold)
    results = calculator.compute_with_predictions(dataset, use_val_fold=True)

    slide_embeddings = results["embeddings"]
    labels = results["labels"]
    predictions = results["predictions"]
    probabilities = results["probabilities"]
    indices = results["indices"]

    print(f"\nSlide embeddings shape: {slide_embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    # Print prediction summary
    accuracy = (predictions == labels).mean()
    print(f"Accuracy: {accuracy:.4f}")

<<<<<<< HEAD
            labels.append(label)
            slide_embeddings.append(s_e)

            # h5_path = str(dataset.h5_files[idx])
            # print(f"hdf5 path:{h5_path}")
            # previewer = PreviewAttention(size=64, model_name=encoder_name)
            # img = previewer(h5_path, attention_scores=att.flatten())
            # preview_path = os.path.join(image_dir_full, "preview", f"{h5_path}_preview.jpg")
            # img.save(preview_path)
            # print(f"    Saved preview: {preview_path}")
    
    slide_embeddings = np.asarray(slide_embeddings)
    labels = np.asarray(labels)
    print(f"\nslide embeddings shape: {slide_embeddings.shape}")
=======
    # # Preview generation (optional - uncomment if needed)
    # print("\n" + "=" * 50)
    # print("Generating attention previews")
    # print("=" * 50)
    # previewer = PreviewAttention(size=64, model_name=encoder_name)
    # preview_dir = os.path.join(image_dir_full, "preview")
    # os.makedirs(preview_dir, exist_ok=True)
    #
    # for fold_idx in range(calculator.predictor.num_folds):
    #     _, val_indices = calculator.predictor.get_fold_indices(fold_idx)
    #     for idx in val_indices:
    #         x, label = dataset[idx]
    #         result = calculator.compute(x, fold_idx)
    #         att = result["attention"]
    #         h5_path = str(dataset.h5_files[idx])
    #         img = previewer(h5_path, attention_scores=att.numpy())
    #         preview_path = os.path.join(preview_dir, f"sample_{idx}_preview.jpg")
    #         img.save(preview_path)
    #         print(f"  Saved: {preview_path}")
>>>>>>> 6333278 (update:)
        
    # UMAP visualization
    print("\n" + "=" * 50)
    print("UMAP Visualization")
    print("=" * 50)

    umap_model = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.3, random_state=42)
    slide_umap = umap_model.fit_transform(slide_embeddings)

    plt.figure(figsize=(10, 10))

    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        mask = labels == i
        plt.scatter(slide_umap[mask, 0], slide_umap[mask, 1], label=f"Class {i}", alpha=0.7)

    plt.legend()
    plt.title("UMAP Visualization of Slide Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)

    umap_path = os.path.join(image_dir_full, "umap.jpeg")
    plt.savefig(umap_path)
    print(f"Saved UMAP plot: {umap_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
