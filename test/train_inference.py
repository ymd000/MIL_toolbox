from mil_toolbox.data import DummyWSIDataset, WSIDataset
from mil_toolbox.models import MILModel
from mil_toolbox.train import CrossValidationTrainer
from mil_toolbox.inference import AttentionAggregator

import os
from pathlib import Path
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

    for i in range(len(dataset)):
        wsi, label = dataset[i]
        print(f"WSI {i}: {wsi.shape}, Label {i}: {label}")

    model_name = "abmil"
    model_config = f'{model_name}.base.{encoder_name}.none'

    model_kwargs = {
        "num_classes": 2,
        "model_config": model_config
    }

    num_fold = 5

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
    print("Inference Test with AttentionAggregator")
    print("=" * 50)

    aggregator = AttentionAggregator(
        model_class=MILModel,
        model_kwargs=model_kwargs,
        output_dir="./outputs",
        device="cpu"
    )

    aggregator.load_models(checkpoint_name="best")

    slide_embeddings = []
    labels = []

    for fold_idx in range(num_fold):
        fold_info = aggregator.fold_manager.get_fold(fold_idx)
        print(f"\n--- Fold {fold_idx} ---")
        print(f"Val indices: {fold_info.val_indices[:5]}...")

        for idx in fold_info.val_indices:
            x, label = dataset[idx]
            result = aggregator.predict_with_attention(x, fold_idx)
            probs = result['probs']
            pred_class = probs.argmax().item()
            pred_prob = probs[pred_class].item()
            att = result['attention']
            att_info = f"shape={att.shape}" if att is not None else "None"
            s_e = result['slide_embedding'] 
            print(f"  Sample {idx}: label={label}, pred={pred_class}, prob={pred_prob:.4f}, attention {att_info}, slide embedding shape={s_e.shape}")

            labels.append(label)
            slide_embeddings.append(s_e)
    
    slide_embeddings = np.asarray(slide_embeddings)
    labels = np.asarray(labels)
    print(f"\nslide embeddings shape: {slide_embeddings.shape}")
        
    umap_model = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.3, random_state=42)
    slide_umap = umap_model.fit_transform(slide_embeddings)
    
    plt.figure(figsize=(10, 10))

    for i in range(2):
        indices = labels == i
        plt.scatter(slide_umap[indices, 0], slide_umap[indices, 1], label=str(i), alpha=0.7)

    plt.legend()
    plt.title("UMAP Visualization")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)
    
    umap_path = os.path.join(image_dir_full,"umap.jpeg")
    plt.savefig(umap_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
