import csv
from pathlib import Path
from mil_toolbox.data import DummyWSIDataset
from mil_toolbox.data.datasets.wsi import WSIDataset as HDF5WSIDataset


def load_labels_from_csv(csv_path: str, id_col: str = "slide_id", label_col: str = "label") -> dict:
    """
    CSVファイルからラベル情報を読み込む

    Args:
        csv_path: CSVファイルのパス
        id_col: スライドIDのカラム名
        label_col: ラベルのカラム名

    Returns:
        ファイル名(拡張子なし)をキー、ラベルを値とする辞書
    """
    label_dict = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_id = row[id_col]
            label = int(row[label_col])
            label_dict[slide_id] = label
    return label_dict


def test_dummy_dataset():
    """DummyWSIDatasetのテスト"""
    print("=== DummyWSIDataset Test ===")
    dataset = DummyWSIDataset(num_wsi=100)
    print(f"Dataset size: {len(dataset)}")

    for i in range(min(5, len(dataset))):
        wsi, label = dataset[i]
        print(f"WSI {i}: shape={wsi.shape}, label={label}")


def test_hdf5_dataset(data_dir: str, csv_path: str):
    """HDF5WSIDatasetのテスト"""
    print("\n=== HDF5WSIDataset Test ===")

    # CSVからラベル情報を読み込み
    label_dict = load_labels_from_csv(csv_path)
    print(f"Loaded {len(label_dict)} labels from CSV")

    # データセットを作成
    dataset = HDF5WSIDataset(data_dir=data_dir, label_dict=label_dict)
    print(f"Dataset size: {len(dataset)}")

    for i in range(min(5, len(dataset))):
        embeddings, label = dataset[i]
        print(f"WSI {i}: shape={embeddings.shape}, label={label}")


def main():
    # DummyWSIDatasetのテスト
    test_dummy_dataset()

    # HDF5WSIDatasetのテスト（パスを指定して実行する場合）
    # data_dir = "data/embedding"
    # csv_path = "data/labels.csv"
    # if Path(data_dir).exists() and Path(csv_path).exists():
    #     test_hdf5_dataset(data_dir, csv_path)


if __name__ == "__main__":
    main()
