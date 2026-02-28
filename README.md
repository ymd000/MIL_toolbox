# MIL_toolbox

病理全スライド画像 (WSI) を対象とした Multiple Instance Learning (MIL) の訓練・推論・可視化ライブラリ。

## 機能

- **Cross-Validation 訓練** - Stratified K-Fold による交差検証、チェックポイント・ログを自動保存
- **スライド埋め込み計算** - attention 重み付き和、最近傍パッチ選択など複数の戦略
- **可視化** - UMAP、混同行列

## インストール

```bash
uv add git+https://github.com/ymd000/MIL_toolbox.git
```

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

results = calculator.compute_and_save(dataset, use_val_fold=True)
```

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
calculator = SlideEmbeddingCalculator(..., output_dir="./outputs")           # 最新
calculator = SlideEmbeddingCalculator(..., output_dir="./outputs", version=0) # 指定
```

## HDF5 データ構造

```
sample.h5
├── {encoder}/
│   └── features                 # パッチ特徴ベクトル (N, feature_dim)
└── slide_embedding/
    └── {method}/
        ├── embedding            # スライド埋め込み (feature_dim,)
        ├── attention            # attention 重み (N,)
        └── probabilities        # 予測確率 (num_classes,)
```

## モデル設定

モデルは設定文字列で指定する。

```python
model_config = "{model}.{variant}.{encoder}.{pooling}"
# 例: "abmil.base.uni.none"
```

| 項目 | 対応値 |
|------|--------|
| model | `abmil` |
| encoder | `uni`, `gigapath`, `virchow2` |

## スライド埋め込み戦略

| 戦略 | モデル不要 | 説明 |
|------|-----------|------|
| `abmil` (デフォルト) | | attention 重み付き和 |
| `nearest_cosine` | ✓ | 平均に最も近いパッチ（コサイン類似度） |
| `nearest_euclidean` | ✓ | 平均に最も近いパッチ（ユークリッド距離） |
| `attention_top` | | attention 最大のパッチ |
| `attention_nearest_cosine` | | attention 重み付き和に最も近いパッチ（コサイン） |
| `attention_nearest_euclidean` | | attention 重み付き和に最も近いパッチ（ユークリッド） |
| `attention_filtered_nearest_cosine` | | 上位 attention パッチの平均に最も近いパッチ（コサイン） |
| `attention_filtered_nearest_euclidean` | | 上位 attention パッチの平均に最も近いパッチ（ユークリッド） |

## 環境構築

```bash
uv sync
source .venv/bin/activate
```

## 外部依存関係

- **mil-lab**: MIL モデル実装 (github.com/ymd000/MIL-Lab)
- **wsi-toolbox**: WSI 処理、特徴抽出 (github.com/technoplasm/wsi-toolbox)
