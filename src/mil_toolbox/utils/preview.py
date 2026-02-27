"""Attention preview generation utilities for MIL analysis."""

from pathlib import Path

import h5py
import numpy as np
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


def save_selected_patch_images(
    data: dict,
    method_name: str,
    output_dir: str | Path,
    patch_size: int = 256,
) -> None:
    """Save the selected representative patch image for each slide.

    Reads selected_index stored by SlideEmbeddingCalculator and extracts
    the corresponding patch image. Uses H5 cache if available, otherwise
    falls back to the original WSI file in the same directory as the H5.

    Args:
        data: dict with keys 'h5_paths' and 'case_names'
        method_name: Method name used when saving embeddings
                     (e.g., "nearest_cosine", "abmil_attention_top")
        output_dir: Directory to save patch images
        patch_size: Patch size used in wsi_toolbox cache
    """
    print("\n" + "=" * 50)
    print("Saving Selected Patch Images")
    print("=" * 50)

    output_dir = Path(output_dir)
    patch_dir = output_dir / "selected_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

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

            # If cache/patch_size/patches exists, read the patch image directly from H5.
            # Otherwise, fall back to on-demand extraction from the original WSI file
            # (wsi_toolbox searches for the WSI in the same directory as the H5).
            if cache_patches_path in f:
                patch_array = f[cache_patches_path][selected_index]
                img = Image.fromarray(patch_array)
                coord = None
            else:
                coord = tuple(int(v) for v in f[cache_coords_path][selected_index])
                img = None

        if img is None:
            reader = get_patch_reader(str(h5_path), patch_size=patch_size)
            patch_array = reader.get_patch_by_coord(coord)
            img = Image.fromarray(patch_array)

        out_path = patch_dir / f"{case_name}_{method_name}.jpeg"
        img.save(out_path)
        print(f"  Saved: {out_path}")

    print(f"\nSelected patches saved to: {patch_dir}")
