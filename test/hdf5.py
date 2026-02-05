import h5py
import numpy as np
import os
import csv
from pathlib import Path


FEATURE_DIMS = {
    "uni": 1024,
    "gigapath": 1536,
    "virchow2": 2560,
}


def create_dummy_hdf5_dataset(
    output_dir: str = "test/data/embedding",
    label_csv_path: str = "test/data/labels.csv",
    num_wsi: int = 100,
    model: str = "uni",
    min_patches: int = 50,
    max_patches: int = 500,
    random_seed: int = 42
):
    """
    ダミーのHDF5ファイルとラベルCSVを生成する

    Args:
        output_dir: HDF5ファイルの出力ディレクトリ
        label_csv_path: ラベルCSVの出力パス
        num_wsi: 生成するWSI数
        model: 
        min_patches: 最小パッチ数
        max_patches: 最大パッチ数
        random_seed: 乱数シード
    """
    np.random.seed(random_seed)

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(Path(label_csv_path).parent, exist_ok=True)
    
    # model
    feature_dim = FEATURE_DIMS.get(model)

    # ラベル情報を格納するリスト
    label_records = []

    for i in range(num_wsi):
        case_id = f"slide_{i:04d}"
        num_patches = np.random.randint(min_patches, max_patches + 1)
        label = np.random.randint(0, 2)

        # ダミーの特徴量を生成
        features = np.random.randn(num_patches, feature_dim).astype(np.float32)

        # HDF5ファイルに保存
        h5_path = os.path.join(output_dir, f"{case_id}.h5")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset(f'{model}/features', data=features)

        label_records.append({'case_id': case_id, 'label': label})
        print(f"Created {h5_path}: shape={features.shape}, label={label}")

    # ラベルCSVを保存
    with open(label_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'label'])
        writer.writeheader()
        writer.writerows(label_records)

    print(f"\nCreated {num_wsi} HDF5 files in {output_dir}")
    print(f"Labels saved to {label_csv_path}")


if __name__ == "__main__":
    create_dummy_hdf5_dataset()
