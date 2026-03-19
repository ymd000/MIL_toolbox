"""LinearProbeModel + EmbeddingDataset の動作確認テスト。

実行方法:
    python test/linear_train.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from mil_toolbox.data import EmbeddingDataset
from mil_toolbox.models import LinearProbeModel
from mil_toolbox.train import CrossValidationTrainer
import lightning as L


# --------------------------------------------------
# ダミーデータ生成
# --------------------------------------------------
def make_dummy_embeddings(n: int = 40, dim: int = 128, num_classes: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    labels = rng.integers(0, num_classes, size=n).tolist()
    case_names = [f"case_{i:03d}" for i in range(n)]
    return embeddings, labels, case_names


# --------------------------------------------------
# テスト 1: EmbeddingDataset の基本動作
# --------------------------------------------------
def test_embedding_dataset():
    print("=" * 50)
    print("Test 1: EmbeddingDataset")
    print("=" * 50)

    embeddings, labels, case_names = make_dummy_embeddings(n=20, dim=64)
    ds = EmbeddingDataset(embeddings, labels, case_names)

    assert len(ds) == 20
    assert ds.embedding_dim == 64
    assert ds.num_classes == 2

    x, y = ds[0]
    assert x.shape == (64,), f"Expected (64,), got {x.shape}"
    assert isinstance(y, int)

    print(f"  len={len(ds)}, embedding_dim={ds.embedding_dim}, num_classes={ds.num_classes}")
    print(f"  item[0]: x.shape={x.shape}, y={y}")

    # from_results テスト（indices あり → 並べ直し確認）
    indices = list(range(19, -1, -1))  # 逆順
    results = {
        "embeddings": np.array(embeddings)[indices],
        "labels": np.array(labels)[indices],
        "indices": indices,
        "case_names": [case_names[i] for i in indices],
    }
    ds2 = EmbeddingDataset.from_results(results, reindex=True)
    assert len(ds2) == 20
    # 並べ直し後は元の順序に戻っているはず
    assert np.allclose(ds2._embeddings.numpy(), np.array(embeddings))
    print("  from_results (reindex=True): OK")

    # from_results テスト（indices なし）
    results_no_idx = {"embeddings": np.array(embeddings), "labels": np.array(labels)}
    ds3 = EmbeddingDataset.from_results(results_no_idx, reindex=True)
    assert len(ds3) == 20
    print("  from_results (no indices): OK")

    print("  Test 1 PASSED\n")


# --------------------------------------------------
# テスト 2: LinearProbeModel の forward 確認
# --------------------------------------------------
def test_linear_probe_forward():
    print("=" * 50)
    print("Test 2: LinearProbeModel forward")
    print("=" * 50)

    model = LinearProbeModel(embedding_dim=64, num_classes=3, lr=1e-3, dropout=0.1)
    x = torch.randn(8, 64)
    out = model(x)

    assert "logits" in out, "logits キーが見つかりません"
    assert "attention" in out, "attention キーが見つかりません"
    assert out["logits"].shape == (8, 3), f"Expected (8,3), got {out['logits'].shape}"
    assert out["attention"] is None

    print(f"  logits.shape={out['logits'].shape}, attention={out['attention']}")
    print("  Test 2 PASSED\n")


# --------------------------------------------------
# テスト 3: DataLoader 経由の学習ステップ確認
# --------------------------------------------------
def test_training_loop():
    print("=" * 50)
    print("Test 3: Training loop (2 epochs)")
    print("=" * 50)

    embeddings, labels, _ = make_dummy_embeddings(n=40, dim=128, num_classes=2)
    ds = EmbeddingDataset(embeddings, labels)

    n_val = 8
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = LinearProbeModel(embedding_dim=128, num_classes=2, lr=1e-3)

    trainer = L.Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("  Training completed.")
    print("  Test 3 PASSED\n")


# --------------------------------------------------
# テスト 4: CrossValidationTrainer + collate_fn=None
# --------------------------------------------------
def test_cross_validation_trainer():
    print("=" * 50)
    print("Test 4: CrossValidationTrainer with EmbeddingDataset (collate_fn=None)")
    print("=" * 50)

    embeddings, labels, _ = make_dummy_embeddings(n=20, dim=64, num_classes=2)
    ds = EmbeddingDataset(embeddings, labels)

    cv_trainer = CrossValidationTrainer(
        model_class=LinearProbeModel,
        model_kwargs={"embedding_dim": 64, "num_classes": 2},
        dataset=ds,
        num_fold=4,
        output_dir="./outputs/linear_probe_test",
        max_epochs=2,
        batch_size=4,
        collate_fn=None,  # EmbeddingDataset はデフォルトコレートで OK
    )

    results = cv_trainer.run()
    assert len(results) == 4
    for r in results:
        assert "fold_idx" in r
        assert "best_val_loss" in r
    best_losses = [r["best_val_loss"] for r in results]
    print(f"  Best val losses per fold: {best_losses}")
    print("  Test 4 PASSED\n")


# --------------------------------------------------
# テスト 5: existing_fold_dir によるfold再利用
# --------------------------------------------------
def test_existing_fold_dir():
    print("=" * 50)
    print("Test 5: CrossValidationTrainer with existing_fold_dir")
    print("=" * 50)

    embeddings, labels, _ = make_dummy_embeddings(n=20, dim=64, num_classes=2)
    ds = EmbeddingDataset(embeddings, labels)

    # まず通常実行して fold を作成
    cv1 = CrossValidationTrainer(
        model_class=LinearProbeModel,
        model_kwargs={"embedding_dim": 64, "num_classes": 2},
        dataset=ds,
        num_fold=4,
        output_dir="./outputs/linear_probe_fold_src",
        max_epochs=1,
        batch_size=4,
        collate_fn=None,
    )
    cv1.run()
    fold_src_dir = cv1.fold_manager.output_dir
    print(f"  Source fold dir: {fold_src_dir}")

    # existing_fold_dir を指定して再利用
    cv2 = CrossValidationTrainer(
        model_class=LinearProbeModel,
        model_kwargs={"embedding_dim": 64, "num_classes": 2},
        dataset=ds,
        num_fold=4,
        output_dir="./outputs/linear_probe_fold_reuse",
        max_epochs=1,
        batch_size=4,
        collate_fn=None,
        existing_fold_dir=fold_src_dir,
    )
    results2 = cv2.run()
    assert len(results2) == 4
    print(f"  Reused fold dir: {cv2.fold_manager.output_dir}")
    print("  Test 5 PASSED\n")


if __name__ == "__main__":
    test_embedding_dataset()
    test_linear_probe_forward()
    test_training_loop()
    test_cross_validation_trainer()
    test_existing_fold_dir()
    print("All tests passed!")
