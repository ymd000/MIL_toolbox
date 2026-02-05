"""UMAP visualization utilities for MIL analysis."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def plot_umap(
    data: dict,
    output_path: str | Path,
    class_names: dict | None = None,
    n_neighbors: int = 5,
    min_dist: float = 0.01,
    random_state: int = 42,
    figsize: tuple = (12, 10),
    annotate: bool = True,
    show_misclassified: bool = True,
    title: str | None = None,
) -> None:
    """Generate UMAP visualization of slide embeddings.

    Args:
        data: dict with keys 'embeddings', 'labels', 'predictions' (optional),
              'case_names' (optional)
        output_path: Path to save the plot
        class_names: Mapping from label index to display name
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed for reproducibility
        figsize: Figure size
        annotate: Whether to annotate points with case names
        show_misclassified: Whether to highlight misclassified samples
        title: Plot title. If None, uses default.
    """
    if not HAS_UMAP:
        raise ImportError("umap-learn is required for UMAP visualization. Install with: pip install umap-learn")

    embeddings = data["embeddings"]
    labels = data["labels"]
    predictions = data.get("predictions")
    case_names = data.get("case_names")

    if class_names is None:
        class_names = {i: f"Class {i}" for i in np.unique(labels)}

    print("\n" + "=" * 50)
    print("UMAP Visualization")
    print("=" * 50)

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    slide_umap = umap_model.fit_transform(embeddings)

    plt.figure(figsize=figsize)

    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        mask = labels == i
        label_name = class_names.get(i, f"Class {i}")
        plt.scatter(
            slide_umap[mask, 0],
            slide_umap[mask, 1],
            label=label_name,
            alpha=0.7,
            s=100,
        )

    # Draw red circles around misclassified points
    if show_misclassified and predictions is not None:
        misclassified = predictions != labels
        if np.any(misclassified):
            plt.scatter(
                slide_umap[misclassified, 0],
                slide_umap[misclassified, 1],
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                s=150,
                label="Misclassified",
            )

    # Annotate each point with case name
    if annotate and case_names is not None:
        for j, (x, y) in enumerate(slide_umap):
            plt.annotate(
                case_names[j],
                (x, y),
                fontsize=7,
                alpha=0.8,
                xytext=(3, 3),
                textcoords="offset points",
            )

    plt.legend(loc="best", fontsize=10)
    plt.title(title or "UMAP Visualization of Slide Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"Saved UMAP plot: {output_path}")
