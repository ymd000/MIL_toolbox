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
)
from .umap import plot_umap
from .preview import PreviewAttention, generate_attention_previews

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
    "PreviewAttention",
    "generate_attention_previews",
]
