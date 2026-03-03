# MIL_toolbox

病理全スライド画像 (WSI) を対象とした Multiple Instance Learning (MIL) の訓練・推論・可視化ライブラリ。

## 機能

- **Cross-Validation 訓練** - Stratified K-Fold による交差検証、チェックポイント・ログを自動保存
- **スライド埋め込み計算** - attention 重み付き和、最近傍パッチ選択など複数の手法
- **可視化** - UMAP、混同行列、attention プレビュー

## インストール

```bash
uv add git+https://github.com/ymd000/MIL_toolbox.git
```

## 処理フロー

```
HDF5ファイル
└── {encoder}/features      # 事前計算済み patch embeddings (N, D)
                                    ↓ WSIDataset
                            patch embeddings (N, D)
                                    ↓ MILModel (ABMIL等)
                            attention weights (N,)
                                    ↓ SlideEmbeddingCalculator
                            slide embedding (D,)
                                    ↓ save_to_hdf5
HDF5ファイル
└── slide_embedding/
    └── {method}/embedding  # スライド埋め込み (D,)
```

**訓練時**: `mil_collate_fn` でパッチ数の異なるスライドをパディングしてバッチ化。
**推論時**: 1サンプルずつそのまま入力（パディングなし）。

## 使い方

### 訓練

```python
from mil_toolbox.data import WSIDataset
from mil_toolbox.models import MILModel
from mil_toolbox.train import CrossValidationTrainer

dataset = WSIDataset("data/embedding", encoder_name="uni", csv_path="data/labels.csv")

trainer = CrossValidationTrainer(
    model_class=MILModel,
    model_kwargs={"num_classes": 2, "model_config": "abmil.base.uni.none"},
    dataset=dataset,
    num_fold=5,
    output_dir="./outputs",
    max_epochs=50,
)
trainer.run()
```

### 推論

```python
from mil_toolbox.inference import SlideEmbeddingCalculator

calculator = SlideEmbeddingCalculator(
    model_class=MILModel,
    model_kwargs={"num_classes": 2, "model_config": "abmil.base.uni.none"},
    output_dir="./outputs",   # version="latest" がデフォルト
    device="auto",
)
calculator.load_models(checkpoint_name="best")

# attention 重み付き和（デフォルト）
results = calculator.compute_and_save(dataset, use_val_fold=True)

# 手法を指定する場合
results = calculator.compute_and_save(dataset, method="abmil_top", use_val_fold=True)
results = calculator.compute_and_save(dataset, method="nearest_cosine")
```

`compute_and_save` の戻り値：

```python
{
    "embeddings":       np.ndarray,        # (N, D)
    "labels":           np.ndarray,        # (N,)
    "predictions":      np.ndarray | None, # abmil のみ
    "probabilities":    np.ndarray | None, # abmil のみ
    "selected_indices": list | None,       # パッチ選択系手法のみ
    "indices":          list,              # データセット内インデックス
}
```

### HDF5 からの読み込み

```python
# 1サンプル
data = SlideEmbeddingCalculator.load_from_hdf5("sample.h5", method_name="abmil")

# データセット全体
data = SlideEmbeddingCalculator.load_dataset_embeddings(
    data_dir="data/embedding",
    method_name="abmil_top",
    csv_path="data/labels.csv",
)
```

### 可視化

```python
from mil_toolbox.utils import (
    compute_metrics_from_results,
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_umap,
    generate_attention_previews,
)

# metrics
metrics = compute_metrics_from_results(results, positive_class=1)

# 混同行列
cm = compute_confusion_matrix(results["labels"], results["predictions"])
plot_confusion_matrix(cm, output_path="cm.jpeg", class_names={0: "A", 1: "B"})

# UMAP
plot_umap(results, output_path="umap.jpeg", class_names={0: "A", 1: "B"})

# attention プレビュー（wsi_toolbox が必要）
data = SlideEmbeddingCalculator.load_dataset_embeddings(...)
generate_attention_previews(data, output_dir="./outputs", encoder_name="uni")
```

## スライド埋め込み手法

`compute_and_save(dataset, method=...)` の `method` に指定する文字列。
`{model}` はモデル設定から自動抽出（例: `"abmil.base.uni.none"` → `"abmil"`）。

| method | モデル必要 | 説明 |
|--------|-----------|------|
| `{model}` (デフォルト) | ✓ | attention 重み付き和 |
| `{model}_top` | ✓ | attention 最大のパッチ1枚 |
| `{model}_nearest_cosine` | ✓ | attention 重み付き和に最も近いパッチ（コサイン） |
| `{model}_nearest_euclidean` | ✓ | attention 重み付き和に最も近いパッチ（ユークリッド） |
| `{model}_filtered_nearest_cosine` | ✓ | 上位 attention パッチの平均に最も近いパッチ（コサイン） |
| `{model}_filtered_nearest_euclidean` | ✓ | 上位 attention パッチの平均に最も近いパッチ（ユークリッド） |
| `nearest_cosine` | | 全パッチ平均に最も近いパッチ（コサイン） |
| `nearest_euclidean` | | 全パッチ平均に最も近いパッチ（ユークリッド） |

`predictions` / `probabilities` を返すのは `{model}`（デフォルト）のみ。
パッチ選択系手法は `selected_indices` を返す。

## 出力ディレクトリ構造

`run()` を呼ぶたびに `version_X` が自動生成される。

```
outputs/
└── version_0/
    ├── config.yaml          # タイムスタンプ・ハイパーパラメータ
    ├── fold_indices.csv     # fold 分割情報
    ├── fold_0/
    │   ├── checkpoints/
    │   │   ├── best.ckpt
    │   │   └── last.ckpt
    │   └── logs/
    │       └── metrics.csv
    └── fold_1/
        └── ...
```

推論時はバージョンを指定できる。

```python
calculator = SlideEmbeddingCalculator(..., output_dir="./outputs")            # 最新
calculator = SlideEmbeddingCalculator(..., output_dir="./outputs", version=0) # 指定
```

## HDF5 データ構造

```
sample.h5
├── {encoder}/
│   └── features                 # patch embeddings (N, D)
└── slide_embedding/
    ├── abmil/
    │   ├── embedding            # スライド埋め込み (D,)
    │   ├── attention            # attention 重み (N,)
    │   └── probabilities        # 予測確率 (num_classes,)
    ├── abmil_top/
    │   ├── embedding            # (D,)
    │   ├── attention            # (N,)
    │   └── [selected_index]     # 選択パッチのインデックス (attr)
    └── nearest_cosine/
        ├── embedding            # (D,)
        └── [selected_index]     # (attr)
```

## モデル設定

```python
model_config = "{model}.{variant}.{encoder}.{pooling}"
# 例: "abmil.base.uni.none"
```

| 項目 | 対応値 |
|------|--------|
| model | `abmil` |
| encoder | `uni`, `gigapath`, `virchow2` |

## 環境構築

```bash
uv sync
source .venv/bin/activate
```

## 外部依存関係

- **mil-lab**: MIL モデル実装 (github.com/ymd000/MIL-Lab)
- **wsi-toolbox**: WSI 処理、特徴抽出 (github.com/technoplasm/wsi-toolbox)
