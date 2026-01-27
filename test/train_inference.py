from mil_toolbox.data import DummyWSIDataset, MILDataModule
from mil_toolbox.models import MILModel
from mil_toolbox.train import CrossValidationTrainer
from mil_toolbox.inference import AttentionAggregator


def main():
    dataset = DummyWSIDataset(num_wsi=100)
    print("Dataset created:")

    for i in range(len(dataset)):
        wsi, label = dataset[i]
        print(f"WSI {i}: {wsi.shape}, Label {i}: {label}")

    model_name = "abmil"
    model_config = f'{model_name}.base.uni.none'

    model_kwargs = {
        "num_classes": 2,
        "model_config": model_config
    }

    num_fold = 5

    print("\n" + "=" * 50)
    print("Train with CrossValidationTrainer")
    print("=" * 50)

    trainer = CrossValidationTrainer(
        model_class=MILModel,
        model_kwargs=model_kwargs,
        dataset=dataset,
        num_fold=num_fold,
        output_dir="./outputs"
    )
    trainer.run()
        
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

    for fold_idx in range(num_fold):
        fold_info = aggregator.fold_manager.get_fold(fold_idx)
        print(f"\n--- Fold {fold_idx} ---")
        print(f"Val indices: {fold_info.val_indices[:5]}...")  # 最初の5つ

        for idx in fold_info.val_indices[:3]:  # 各foldから3サンプル
            x, label = dataset[idx]
            result = aggregator.predict_with_attention(x, fold_idx)
            probs = result['probs'].squeeze()
            pred_class = probs.argmax().item()
            pred_prob = probs[pred_class].item()
            att = result['attention']
            att_info = f"shape={att.squeeze().shape}" if att is not None else "None"
            print(f"  Sample {idx}: label={label}, pred={pred_class}, prob={pred_prob:.4f}, attention {att_info}")

    print("\nDone.")


if __name__ == "__main__":
    main()
