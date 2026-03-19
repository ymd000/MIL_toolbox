# MIL_toolbox

病理全スライド画像 (WSI) を対象とした Multiple Instance Learning (MIL) の訓練・推論・可視化ライブラリ。

## 機能

- **Cross-Validation 訓練** - Stratified K-Fold による交差検証、チェックポイント・ログを自動保存
- **スライド埋め込み計算** - attention 重み付き和、最近傍パッチ選択など複数の手法
- **TITAN 推論** - TITAN Foundation Model によるスライドレベル埋め込み計算（fold なし）
- **Linear Probe** - スライド埋め込みに対する線形分類器の訓練・推論
- **可視化** - UMAP、混同行列、attention プレビュー

## インストール

```bash
uv add git+https://github.com/ymd000/MIL_toolbox.git
```

## 処理フロー

### ABMIL（MIL モデル）

```
HDF5ファイル
└── {encoder}/features      # 事前計算済み patch embeddings (N, D)
                                    ↓ WSIDataset
                            patch embeddings (N, D)
                                    ↓ MILModel (ABMIL等)
                            attention weights (N,)
                                    ↓ SlideEmbeddingCalculator
                            slide embedding (D,)
                                    ↓ _save_to_hdf5
HDF5ファイル
└── slide_embedding/
    └── {method}/embedding  # スライド埋め込み (D,)
```

**訓練時**: `mil_collate_fn` でパッチ数の異なるスライドをパディングしてバッチ化。
**推論時**: 1サンプルずつそのまま入力（パディングなし）。

### TITAN（Foundation Model）

```
HDF5ファイル
├── {encoder}/features      # patch embeddings (N, D)
└── {encoder}/coordinates   # level 0 基準座標 (N, 2)
                                    ↓ TITANAggregator
                            slide embedding (D,)
                                    ↓ _save_to_hdf5
HDF5ファイル
└── slide_embedding/
    └── titan/embedding     # スライド埋め込み (D,)
```

### Linear Probe（スライド埋め込みの線形分類）

```
slide embeddings (N, D)
        ↓ EmbeddingDataset
        ↓ CrossValidationTrainer (existing_fold_dir で fold を再利用)
LinearProbeModel
        ↓ MILPredictor
predictions / confusion matrix
```

---

## 使い方

### 訓練

```python
from mil_toolbox.data import WSIDataset
from mil_toolbox.models import MILModel
from mil_toolbox.train import CrossValidationTrainer

dataset = WSIDataset("data/embedding", model="uni", csv_path="data/labels.csv")

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

`CrossValidationTrainer` の主要オプション:

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `collate_fn` | `mil_collate_fn` | DataLoader の collate 関数。`EmbeddingDataset` 使用時は `None` を指定 |
| `existing_fold_dir` | `None` | 既存の fold 分割を再利用する場合に version_X ディレクトリを指定 |

### 推論（ABMIL）

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
    "attentions":       list | None,
    "indices":          list,              # データセット内インデックス
    "h5_paths":         list[Path],
    "case_names":       list[str],
}
```

### 推論（TITAN）

`TITANAggregator` は fold なし・Lightning ckpt なしで動作する推論専用クラス。
モデルのロードはサブクラスで実装する（`transformers` は呼び出し側でインストール）。

```python
from mil_toolbox.inference import TITANAggregator

class MyTITAN(TITANAggregator):
    def _load_model(self):
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            './models/titan',
            trust_remote_code=True,
            local_files_only=True,
        )

dataset = WSIDataset("data/embedding", model="conch15_768", csv_path="data/labels.csv")

aggregator = MyTITAN(
    patch_size_lv0=512,       # level 0 換算パッチサイズ
    encoder_name="conch15_768",   # HDF5 内の features/coordinates キー
    device="auto",
)
aggregator.load_model()

results = aggregator.compute_and_save(dataset, overwrite=False)
```

`patch_size_lv0` の決め方：

| 抽出条件 | patch_size_lv0 |
|---------|---------------|
| level 1 (downsample=2.0) / 256px | 512 |
| level 1 (downsample=2.0) / 512px | 1024 |

`compute_and_save` は `SlideEmbeddingCalculator` と同じキー構成の dict を返す。
`predictions` / `probabilities` / `selected_indices` / `attentions` は常に `None`。

### Linear Probe

スライドレベル埋め込みを入力とする線形分類器。
ABMIL 等で得た埋め込みに対して同じ fold 分割で訓練できる。

```python
from mil_toolbox.data import EmbeddingDataset
from mil_toolbox.models import LinearProbeModel
from mil_toolbox.train import CrossValidationTrainer
from mil_toolbox.inference import MILPredictor

# スライド埋め込みからデータセットを構築
# results は compute_and_save の戻り値
embed_dataset = EmbeddingDataset.from_results(results, reindex=True)
# または直接構築
embed_dataset = EmbeddingDataset(embeddings, labels)

# ABMIL と同じ fold 分割で訓練
lp_trainer = CrossValidationTrainer(
    model_class=LinearProbeModel,
    model_kwargs={"embedding_dim": embed_dataset.embedding_dim, "num_classes": 2},
    dataset=embed_dataset,
    num_fold=5,
    output_dir="./outputs",
    max_epochs=50,
    collate_fn=None,                      # EmbeddingDataset 使用時は None
    existing_fold_dir="./outputs/version_0",  # ABMIL の fold を再利用
)
lp_trainer.run()

# 推論
predictor = MILPredictor(
    model_class=LinearProbeModel,
    model_kwargs={"embedding_dim": embed_dataset.embedding_dim, "num_classes": 2},
    output_dir="./outputs",
    version="latest",
    device="cpu",
)
predictor.load_models()

# 1サンプルの推論（attention=None のため return_attention=False 必須）
x, label = embed_dataset[0]
result = predictor.predict(x, fold_idx=0, return_attention=False)
```

`LinearProbeModel` の引数：

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `embedding_dim` | — | 入力埋め込みの次元数 |
| `num_classes` | — | 出力クラス数 |
| `lr` | `1e-3` | 学習率 |
| `dropout` | `0.0` | Dropout 率 |
| `weight_decay` | `0.0` | Adam の weight_decay |

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

# UMAP（show_misclassified=True で誤分類に赤バツを表示）
plot_umap(
    results,
    output_path="umap.jpeg",
    class_names={0: "A", 1: "B"},
    show_misclassified=True,
)

# attention プレビュー（wsi_toolbox が必要）
data = SlideEmbeddingCalculator.load_dataset_embeddings(...)
generate_attention_previews(data, output_dir="./outputs", encoder_name="uni")
```

---

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

---

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

---

## HDF5 データ構造

```
sample.h5
├── {encoder}/
│   ├── features                 # patch embeddings (N, D)
│   └── coordinates              # level 0 基準座標 (N, 2)  ← TITAN 使用時に必要
└── slide_embedding/
    ├── abmil/
    │   ├── embedding            # スライド埋め込み (D,)
    │   ├── attention            # attention 重み (N,)
    │   └── probabilities        # 予測確率 (num_classes,)
    ├── abmil_top/
    │   ├── embedding            # (D,)
    │   ├── attention            # (N,)
    │   └── [selected_index]     # 選択パッチのインデックス (attr)
    ├── nearest_cosine/
    │   ├── embedding            # (D,)
    │   └── [selected_index]     # (attr)
    └── titan/
        └── embedding            # スライド埋め込み (D,)
```

---

## モデル設定

```python
model_config = "{model}.{variant}.{encoder}.{pooling}"
# 例: "abmil.base.uni.none"
```

| 項目 | 対応値 |
|------|--------|
| model | `abmil` |
| encoder | `uni`, `uni2`, `gigapath`, `virchow2`, `conch15_768` |

対応エンコーダーの特徴量次元：

| encoder | dim |
|---------|-----|
| `uni` | 1024 |
| `uni2` | 1536 |
| `gigapath` | 1536 |
| `virchow2` | 2560 |
| `conch15_768` | 768 |

---

## 環境構築

```bash
uv sync
source .venv/bin/activate
```

## 外部依存関係

- **mil-lab**: MIL モデル実装 (github.com/ymd000/MIL-Lab)
- **wsi-toolbox**: WSI 処理、特徴抽出 (github.com/technoplasm/wsi-toolbox)
- **transformers**: TITAN 使用時は呼び出し側プロジェクトでインストール
