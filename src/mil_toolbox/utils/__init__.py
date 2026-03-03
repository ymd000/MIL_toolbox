"""Utility functions for MIL Toolbox."""

from .plot_training_log import plot_training_metrics, load_metrics
from .metrics import (
    compute_metrics,
    compute_metrics_from_results,
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_confusion_matrix,
    format_metrics_table,
    print_metrics,
    metrics_to_dataframe,
    plot_confusion_matrix,
)
from .umap import plot_umap

__all__ = [
    "plot_training_metrics",
    "load_metrics",
    "compute_metrics",
    "compute_metrics_from_results",
    "compute_binary_metrics",
    "compute_multiclass_metrics",
    "compute_confusion_matrix",
    "format_metrics_table",
    "print_metrics",
    "metrics_to_dataframe",
    "plot_umap",
    "plot_confusion_matrix",
    "PreviewAttention",
    "generate_attention_previews",
    "generate_attention_previews_from_dir",
    "save_selected_patches",
]


def __getattr__(name):
    if name in ("PreviewAttention", "generate_attention_previews", "generate_attention_previews_from_dir", "save_selected_patches"):
        from .preview import PreviewAttention, generate_attention_previews, generate_attention_previews_from_dir, save_selected_patches

        globals()["PreviewAttention"] = PreviewAttention
        globals()["generate_attention_previews"] = generate_attention_previews
        globals()["generate_attention_previews_from_dir"] = generate_attention_previews_from_dir
        globals()["save_selected_patches"] = save_selected_patches
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
