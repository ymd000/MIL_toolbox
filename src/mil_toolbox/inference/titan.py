"""TITAN Foundation Model によるスライドレベル埋め込み計算（推論専用）。"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


class TITANAggregator:
    """TITAN によるスライドレベル埋め込み計算クラス（推論専用）。

    FoldManager / Lightning ckpt / CrossValidationTrainer には依存しない。
    HDF5 保存形式は SlideEmbeddingCalculator と同じ規約に従い、
    ``slide_embedding/{method_name}/embedding`` に保存する。

    モデルのロードは :meth:`_load_model` をサブクラスでオーバーライドして実装する。

    Example::

        class MyTITAN(TITANAggregator):
            def _load_model(self):
                from transformers import AutoModel
                return AutoModel.from_pretrained(
                    './models/titan',
                    trust_remote_code=True,
                    local_files_only=True,
                )

        aggregator = MyTITAN(patch_size_lv0=512, encoder_name="conch15_768")
        aggregator.load_model()
        results = aggregator.compute_and_save(dataset)

    Args:
        patch_size_lv0: level 0 換算のパッチサイズ（ピクセル）。
            例: level 1 (downsample=2.0) / 256px 抽出なら 512、
                level 1 (downsample=2.0) / 512px 抽出なら 1024。
        encoder_name: HDF5 内の patch embedding キー名。
            features     → ``{encoder_name}/features``
            coordinates  → ``{encoder_name}/coordinates``
        device: 使用デバイス。``"auto"`` で GPU があれば CUDA を選択。
        method_name: HDF5 保存時の group 名（``slide_embedding/{method_name}``）。
    """

    def __init__(
        self,
        patch_size_lv0: int,
        encoder_name: str = "conch15_768",
        device: str = "auto",
        method_name: str = "titan",
    ):
        self.patch_size_lv0 = patch_size_lv0
        self.encoder_name = encoder_name
        self.method_name = method_name

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None

    # ------------------------------------------------------------------
    # モデルロード
    # ------------------------------------------------------------------

    def _load_model(self):
        """サブクラスで TITAN モデルを返すようにオーバーライドする。

        Raises:
            NotImplementedError: サブクラスで実装されていない場合。

        Example::

            class MyTITAN(TITANAggregator):
                def _load_model(self):
                    from transformers import AutoModel
                    return AutoModel.from_pretrained(
                        './models/titan',
                        trust_remote_code=True,
                        local_files_only=True,
                    )
        """
        raise NotImplementedError(
            "_load_model() をサブクラスで実装してください。\n"
            "\n"
            "実装例:\n"
            "    class MyTITAN(TITANAggregator):\n"
            "        def _load_model(self):\n"
            "            from transformers import AutoModel\n"
            "            return AutoModel.from_pretrained(\n"
            "                './models/titan',\n"
            "                trust_remote_code=True,\n"
            "                local_files_only=True,\n"
            "            )\n"
        )

    def load_model(self, model=None) -> None:
        """モデルをロードして eval モードにする。

        Args:
            model: 外部から渡すモデルオブジェクト。
                ``None`` の場合は :meth:`_load_model` を呼ぶ。
        """
        if model is not None:
            self.model = model
        else:
            self.model = self._load_model()

        self.model.to(self.device).eval()
        print(f"TITAN model loaded on {self.device}")

    # ------------------------------------------------------------------
    # 単一スライド推論
    # ------------------------------------------------------------------

    def compute(
        self,
        patch_embeddings: torch.Tensor,
        coords: torch.Tensor | np.ndarray,
        patch_size_lv0: int | None = None,
        normalize: bool = False,
    ) -> dict:
        """単一スライドのスライドレベル埋め込みを計算する。

        Args:
            patch_embeddings: 形状 ``(N, D)`` のパッチ埋め込みテンソル。
            coords: 形状 ``(N, 2)`` の level 0 基準座標。Tensor または ndarray。
            patch_size_lv0: level 0 換算のパッチサイズ。
                ``None`` のとき ``self.patch_size_lv0`` を使用。
            normalize: ``True`` のとき出力埋め込みを L2 正規化する。

        Returns:
            dict::

                {
                    "slide_embedding": Tensor(D,),
                    "attention": None,
                    "probs": None,
                    "pred_class": None,
                }
        """
        if self.model is None:
            raise RuntimeError("load_model() を先に呼んでください。")

        if patch_size_lv0 is None:
            patch_size_lv0 = self.patch_size_lv0

        patch_embeddings = patch_embeddings.to(self.device)

        if not isinstance(coords, torch.Tensor):
            coords = torch.from_numpy(np.array(coords))
        coords = coords.to(self.device)

        if self.device.type == "cuda":
            with torch.autocast("cuda", torch.float16), torch.inference_mode():
                slide_embedding = self.model.encode_slide_from_patch_features(
                    patch_embeddings,
                    coords,
                    patch_size_lv0,
                )
        else:
            with torch.inference_mode():
                slide_embedding = self.model.encode_slide_from_patch_features(
                    patch_embeddings,
                    coords,
                    patch_size_lv0,
                )

        # (1, D) → (D,) に正規化
        if slide_embedding.dim() > 1:
            slide_embedding = slide_embedding.squeeze(0)

        slide_embedding = slide_embedding.float().cpu()

        if normalize:
            slide_embedding = slide_embedding / slide_embedding.norm()

        return {
            "slide_embedding": slide_embedding,
            "attention": None,
            "probs": None,
            "pred_class": None,
        }

    # ------------------------------------------------------------------
    # データセット全体推論
    # ------------------------------------------------------------------

    def compute_and_save(
        self,
        dataset,
        normalize: bool = False,
        overwrite: bool = True,
    ) -> dict:
        """データセット全体のスライドレベル埋め込みを計算して HDF5 に保存する。

        Args:
            dataset: ``WSIDataset`` インスタンス。
                ``dataset[idx]`` → ``(Tensor(N, D), int)``
                ``dataset.h5_files[idx]`` → HDF5 ファイルパス
            normalize: 埋め込みを L2 正規化するか否か。
            overwrite: ``False`` のとき既存グループがあるスライドをスキップする。

        Returns:
            dict:
                SlideEmbeddingCalculator.compute_and_save() と同じキー構成::

                    {
                        "embeddings": np.ndarray(N, D),
                        "labels": np.ndarray(N,),
                        "predictions": None,
                        "probabilities": None,
                        "selected_indices": None,
                        "attentions": None,
                        "indices": list[int],
                        "h5_paths": list[Path],
                        "case_names": list[str],
                    }
        """
        embeddings = []
        labels = []
        indices = []
        h5_paths = []
        case_names = []

        for idx in range(len(dataset)):
            h5_path = dataset.h5_files[idx]

            # overwrite=False のとき既存グループがあればスキップ
            if not overwrite:
                group_path = f"{self.encoder_name}/slide_embedding/{self.method_name}"
                with h5py.File(h5_path, "r") as f:
                    if group_path in f:
                        print(f"Skip (already exists): {h5_path.name}")
                        # 既存データを読み込んで結果に含める
                        slide_emb = f[group_path]["embedding"][:]
                        embeddings.append(slide_emb)
                        labels.append(dataset.labels[idx])
                        indices.append(idx)
                        h5_paths.append(h5_path)
                        case_names.append(h5_path.stem)
                        continue

            # patch embedding と座標を取得
            features, label = dataset[idx]

            with h5py.File(h5_path, "r") as f:
                coords = f[f"{self.encoder_name}/coordinates"][:]  # (N, 2)

            result = self.compute(features, coords, normalize=normalize)

            slide_emb = result["slide_embedding"].numpy()

            self._save_to_hdf5(h5_path, slide_emb, overwrite=overwrite)

            embeddings.append(slide_emb)
            labels.append(label)
            indices.append(idx)
            h5_paths.append(h5_path)
            case_names.append(h5_path.stem)

            print(f"Saved slide embedding ({self.method_name}) to {h5_path.name}")

        return {
            "embeddings": np.array(embeddings),
            "labels": np.array(labels),
            "predictions": None,
            "probabilities": None,
            "selected_indices": None,
            "attentions": None,
            "indices": indices,
            "h5_paths": h5_paths,
            "case_names": case_names,
        }

    # ------------------------------------------------------------------
    # HDF5 保存
    # ------------------------------------------------------------------

    def _save_to_hdf5(
        self,
        h5_path: str | Path,
        slide_embedding: np.ndarray,
        overwrite: bool = True,
    ) -> None:
        """スライドレベル埋め込みを HDF5 ファイルに保存する。

        保存パス: ``{encoder_name}/slide_embedding/{method_name}/embedding``

        Args:
            h5_path: 保存先の HDF5 ファイルパス。
            slide_embedding: 保存するスライドレベル埋め込み配列。
            overwrite: ``False`` のとき既存グループがあればスキップする。
        """
        group_path = f"{self.encoder_name}/slide_embedding/{self.method_name}"

        with h5py.File(h5_path, "a") as f:
            if group_path in f:
                if not overwrite:
                    return
                del f[group_path]

            grp = f.create_group(group_path)
            grp.create_dataset("embedding", data=slide_embedding)
