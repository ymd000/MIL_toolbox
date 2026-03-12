"""Attention preview generation utilities for MIL analysis."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from PIL import Image, ImageFont

from wsi_toolbox.commands.preview import BasePreviewCommand
from wsi_toolbox.patch_reader import get_patch_reader
from wsi_toolbox.utils import create_frame, get_platform_font


class PreviewAttention(BasePreviewCommand):
    """Generate thumbnail with attention score visualization.

    Usage:
        previewer = PreviewAttention(size=64)
        img = previewer(hdf5_path='case_id.h5', attention_scores=scores)
    """

    def _prepare(
        self,
        f: h5py.File,
        attention_scores: np.ndarray,
        cmap_name: str = "jet",
    ):
        """Prepare attention visualization data.

        Args:
            f: HDF5 file handle
            attention_scores: Attention scores array (n_patches,)
            cmap_name: Colormap name

        Returns:
            dict with 'scores', 'cmap', and 'font'
        """
        scores = attention_scores.copy()
        s_min, s_max = scores.min(), scores.max()
        scores = (scores - s_min) / (s_max - s_min)

        font = ImageFont.truetype(font=get_platform_font(), size=self.font_size)
        cmap = plt.get_cmap(cmap_name)

        return {"scores": scores, "cmap": cmap, "font": font}

    def _get_frame(self, index: int, data, f: h5py.File):
        """Get frame for attention score at index."""
        score = data["scores"][index]

        if np.isnan(score):
            return None

        color = mcolors.rgb2hex(data["cmap"](score)[:3])
        return create_frame(self.size, color, f"{score:.2f}", data["font"])


def generate_attention_previews(
    data: dict,
    output_dir: str | Path,
    encoder_name: str,
    preview_size: int = 64,
) -> None:
    """Generate attention preview images for all samples.

    H5ファイルとNDPIが同じディレクトリにある場合に使用する。
    H5とNDPIが別ディレクトリの場合は generate_attention_previews_from_dir を使う。

    Args:
        data: dict with keys 'h5_paths', 'attentions', 'case_names'
        output_dir: Directory to save preview images
        encoder_name: Encoder name for preview generation (e.g., "uni", "gigapath")
        preview_size: Size of preview image patches
    """
    print("\n" + "=" * 50)
    print("Generating Attention Previews")
    print("=" * 50)

    output_dir = Path(output_dir)
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    previewer = PreviewAttention(size=preview_size, model_name=encoder_name)

    h5_paths = data["h5_paths"]
    attentions = data["attentions"]
    case_names = data["case_names"]

    for h5_path, attention, case_name in zip(h5_paths, attentions, case_names):
        if attention is None:
            print(f"  Skipping {case_name}: no attention weights.")
            continue

        img = previewer(str(h5_path), attention_scores=attention)
        preview_path = preview_dir / f"{case_name}_preview.jpeg"
        img.save(preview_path)
        print(f"  Saved: {preview_path}")

    print(f"\nPreviews saved to: {preview_dir}")


def generate_attention_previews_from_dir(
    csv_path: str | Path,
    embed_h5_dir: str | Path,
    src_h5_dir: str | Path,
    method_name: str,
    output_dir: str | Path,
    encoder_name: str = "uni",
    preview_size: int = 64,
) -> None:
    """Attentionプレビュー画像を生成する（H5とNDPIが別ディレクトリの場合）。

    embed_h5_dir/{case_id}.h5 の slide_embedding/{method_name}/attention から
    attention スコアを読み、src_h5_dir/{case_id}.h5 をベースにプレビューを生成する。
    src_h5_dir には対応する NDPI ファイルが置かれている必要がある
    （BasePreviewCommand が同ディレクトリから自動探索するため）。

    Args:
        csv_path: case_id, label 列を含むCSVファイルのパス
        embed_h5_dir: スライド埋め込み・attentionを含む HDF5 ディレクトリ
        src_h5_dir: 座標・NDPI ファイルがある HDF5 ディレクトリ
        method_name: メソッド名 (例: "abmil", "abmil_top")
        output_dir: プレビュー画像の保存ディレクトリ
        encoder_name: エンコーダ名 (例: "uni", "gigapath")
        preview_size: プレビューのパッチサイズ
    """
    print("\n" + "=" * 50)
    print(f"Generating Attention Previews: {method_name}")
    print("=" * 50)

    embed_h5_dir = Path(embed_h5_dir)
    src_h5_dir = Path(src_h5_dir)
    output_dir = Path(output_dir)
    preview_dir = output_dir / f"preview_{method_name}"
    preview_dir.mkdir(parents=True, exist_ok=True)

    group_path = f"{encoder_name}/slide_embedding/{method_name}"
    previewer = PreviewAttention(size=preview_size, model_name=encoder_name)
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        case_id = row["case_id"]
        embed_h5 = embed_h5_dir / f"{case_id}.h5"
        src_h5 = src_h5_dir / f"{case_id}.h5"

        if not embed_h5.exists():
            print(f"  Skipping {case_id}: embed HDF5 not found.")
            continue
        if not src_h5.exists():
            print(f"  Skipping {case_id}: src HDF5 not found.")
            continue

        with h5py.File(embed_h5, "r") as f:
            if group_path not in f:
                print(f"  Skipping {case_id}: no embedding for '{method_name}'.")
                continue
            grp = f[group_path]
            if "attention" not in grp:
                print(f"  Skipping {case_id}: no attention weights.")
                continue
            attention = grp["attention"][:]

        img = previewer(str(src_h5), attention_scores=attention)
        preview_path = preview_dir / f"{case_id}.jpeg"
        img.save(preview_path)
        print(f"  Saved: {preview_path}")

    print(f"\nDone. Previews saved to: {preview_dir}")


def save_selected_patch_images(
    data: dict,
    method_name: str,
    output_dir: str | Path,
    patch_size: int = 256,
    ndpi_dir: str | Path | None = None,
) -> None:
    """Save the selected representative patch image for each slide.

    Reads selected_index stored by SlideEmbeddingCalculator and extracts
    the corresponding patch image.

    Priority:
    1. H5 cache (cache/{patch_size}/patches) — reads patch by index directly.
    2. NDPI file matched by case_id from ndpi_dir — extracts patch via coordinate.
    3. If ndpi_dir is None, falls back to WSI auto-discovery by wsi_toolbox.

    Args:
        data: dict with keys 'h5_paths' and 'case_names'
        method_name: Method name used when saving embeddings
                     (e.g., "nearest_cosine", "abmil_top")
        output_dir: Directory to save patch images
        patch_size: Patch size used in wsi_toolbox cache
        ndpi_dir: Directory containing NDPI files. Files are matched to cases
                  by stem (i.e. ``{case_id}.ndpi``). Used when H5 cache is
                  not available.
    """
    print("\n" + "=" * 50)
    print("Saving Selected Patch Images")
    print("=" * 50)

    output_dir = Path(output_dir)
    patch_dir = output_dir / "selected_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    # Build case_id → ndpi path mapping up front
    ndpi_map: dict[str, Path] = {}
    if ndpi_dir is not None:
        for ndpi_path in Path(ndpi_dir).glob("*.ndpi"):
            ndpi_map[ndpi_path.stem] = ndpi_path

    group_path = f"slide_embedding/{method_name}"
    cache_patches_path = f"cache/{patch_size}/patches"
    cache_coords_path = f"cache/{patch_size}/coordinates"

    for h5_path, case_name in zip(data["h5_paths"], data["case_names"]):
        with h5py.File(h5_path, "r") as f:
            if group_path not in f:
                print(f"  Skipping {case_name}: no embedding for method '{method_name}'.")
                continue

            grp = f[group_path]
            if "selected_index" not in grp.attrs:
                print(f"  Skipping {case_name}: no selected_index.")
                continue

            selected_index = int(grp.attrs["selected_index"])

            # Priority 1: use cached patch image directly (fast, index-based)
            if cache_patches_path in f:
                patch_array = f[cache_patches_path][selected_index]
                img = Image.fromarray(patch_array)
                coord = None
            else:
                coord = tuple(int(v) for v in f[cache_coords_path][selected_index])
                img = None

        if img is None:
            # Priority 2: ndpi_dir specified → look up NDPI by case_id
            if ndpi_dir is not None:
                if case_name not in ndpi_map:
                    print(f"  Skipping {case_name}: no NDPI file found in '{ndpi_dir}'.")
                    continue
                wsi_path = str(ndpi_map[case_name])
            else:
                # Priority 3: fall back to wsi_toolbox auto-discovery
                wsi_path = None

            reader = get_patch_reader(str(h5_path), wsi_path=wsi_path, patch_size=patch_size)
            patch_array = reader.get_patch_by_coord(coord)
            img = Image.fromarray(patch_array)

        out_path = patch_dir / f"{case_name}_{method_name}.jpeg"
        img.save(out_path)
        print(f"  Saved: {out_path}")

    print(f"\nSelected patches saved to: {patch_dir}")


def save_selected_patches(
    csv_path: str | Path,
    embed_h5_dir: str | Path,
    ndpi_dir: str | Path,
    method_name: str,
    output_dir: str | Path,
    encoder_name: str = "uni",
    patch_size: int = 256,
) -> None:
    """selected_indexのパッチ画像を保存する。

    embed_h5_dir/{case_id}.h5 の slide_embedding/{method_name} から
    selected_index を読み、{encoder_name}/coordinates[selected_index] の座標で
    ndpi_dir/{case_id}.ndpi からパッチを抽出して保存する。

    Args:
        csv_path: case_id, label 列を含むCSVファイルのパス
        embed_h5_dir: スライド埋め込みを含む HDF5 ディレクトリ
        ndpi_dir: NDPI ファイルのディレクトリ
        method_name: メソッド名 (例: "abmil_top", "abmil_nearest_cosine")
        output_dir: パッチ画像の保存ディレクトリ
        encoder_name: エンコーダ名 (例: "uni", "gigapath") — 座標の読み出し先
                      {encoder_name}/coordinates から取得する
        patch_size: パッチサイズ
    """
    print("\n" + "=" * 50)
    print(f"Saving Selected Patches: {method_name}")
    print("=" * 50)

    embed_h5_dir = Path(embed_h5_dir)
    ndpi_dir = Path(ndpi_dir)
    output_dir = Path(output_dir)
    select_dir = output_dir / "selected_patches"
    select_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = select_dir / f"{method_name}"
    patch_dir.mkdir(parents=True, exist_ok=True)

    group_path = f"{encoder_name}/slide_embedding/{method_name}"
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        case_id = row["case_id"]
        h5_path = embed_h5_dir / f"{case_id}.h5"
        ndpi_path = ndpi_dir / f"{case_id}.ndpi"

        if not h5_path.exists():
            print(f"  Skipping {case_id}: HDF5 not found.")
            continue
        if not ndpi_path.exists():
            print(f"  Skipping {case_id}: NDPI not found.")
            continue

        with h5py.File(h5_path, "r") as f:
            if group_path not in f:
                print(f"  Skipping {case_id}: no embedding for '{method_name}'.")
                continue

            grp = f[group_path]
            if "selected_index" not in grp.attrs:
                print(f"  Skipping {case_id}: no selected_index.")
                continue

            selected_index = int(grp.attrs["selected_index"])
            coord = tuple(int(v) for v in f[f"{encoder_name}/coordinates"][selected_index])

        reader = get_patch_reader(
            str(h5_path), wsi_path=str(ndpi_path), patch_size=patch_size
        )
        patch_array = reader.get_patch_by_coord(coord)
        img = Image.fromarray(patch_array)

        out_path = patch_dir / f"{case_id}.jpeg"
        img.save(out_path)
        print(f"  Saved: {out_path}")

    print(f"\nDone. Patches saved to: {patch_dir}")
