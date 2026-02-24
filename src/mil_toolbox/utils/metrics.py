"""Classification metrics computation, reporting, and visualization utilities."""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: Ground truth labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        num_classes: Number of classes. If None, inferred from data.

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
        where cm[i, j] = count of samples with true label i and predicted label j
    """
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: int = 1,
    class_names: Optional[dict[int, str]] = None,
) -> dict:
    """Compute binary classification metrics.

    Args:
        y_true: Ground truth labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        positive_class: Label considered as positive class (default: 1)
        class_names: Optional dict mapping class indices to string names
                     e.g., {0: "Benign", 1: "Malignant"}

    Returns:
        dict containing:
            - tp, tn, fp, fn: Confusion matrix counts
            - accuracy: Overall accuracy
            - sensitivity: TP / (TP + FN), also known as recall
            - specificity: TN / (TN + FP)
            - precision: TP / (TP + FP)
            - balanced_accuracy: (sensitivity + specificity) / 2
            - roc_approximated: (sensitivity + precision) / 2
            - f1_score: Harmonic mean of precision and sensitivity
            - positive_class: The positive class index
            - class_names: The class names dict (if provided)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Binary conversion
    y_true_bin = (y_true == positive_class).astype(int)
    y_pred_bin = (y_pred == positive_class).astype(int)

    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

    total = tp + tn + fp + fn

    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Sensitivity (Recall) = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Balanced Accuracy = (Sensitivity + Specificity) / 2
    balanced_accuracy = (sensitivity + specificity) / 2

    # ROC Approximated = (Sensitivity + Precision) / 2
    roc_approximated = (sensitivity + precision) / 2

    # F1 Score = 2 * Precision * Sensitivity / (Precision + Sensitivity)
    f1_score = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    # Determine negative class
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    negative_class = [c for c in unique_classes if c != positive_class][0] if len(unique_classes) > 1 else (1 - positive_class)

    result = {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "balanced_accuracy": float(balanced_accuracy),
        "roc_approximated": float(roc_approximated),
        "f1_score": float(f1_score),
        "positive_class": positive_class,
        "negative_class": int(negative_class),
    }

    if class_names is not None:
        result["class_names"] = class_names

    return result


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
    class_names: Optional[dict[int, str]] = None,
) -> dict:
    """Compute multiclass classification metrics.

    For multiclass, metrics are computed per-class and then averaged.

    Args:
        y_true: Ground truth labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        average: Averaging method ("macro", "micro", or "weighted")
        class_names: Optional dict mapping class indices to string names

    Returns:
        dict containing all metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    # Overall accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm) if np.sum(cm) > 0 else 0.0

    # Per-class metrics
    sensitivities = []
    specificities = []
    precisions = []
    supports = []

    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - tp - fn - fp
        support = np.sum(cm[c, :])

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        sensitivities.append(sens)
        specificities.append(spec)
        precisions.append(prec)
        supports.append(support)

    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    precisions = np.array(precisions)
    supports = np.array(supports)

    # Averaging
    if average == "macro":
        sensitivity = np.mean(sensitivities)
        specificity = np.mean(specificities)
        precision = np.mean(precisions)
    elif average == "micro":
        # Micro averaging: sum all TPs, FNs, FPs
        total_tp = np.sum(np.diag(cm))
        total_fn = np.sum(cm) - total_tp
        total_fp = np.sum(cm) - total_tp
        sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        specificity = np.mean(specificities)  # micro doesn't apply well to specificity
    elif average == "weighted":
        total_support = np.sum(supports)
        if total_support > 0:
            weights = supports / total_support
            sensitivity = np.sum(sensitivities * weights)
            specificity = np.sum(specificities * weights)
            precision = np.sum(precisions * weights)
        else:
            sensitivity = specificity = precision = 0.0
    else:
        raise ValueError(f"Unknown average method: {average}")

    balanced_accuracy = (sensitivity + specificity) / 2
    roc_approximated = (sensitivity + precision) / 2
    f1_score = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    result = {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "balanced_accuracy": float(balanced_accuracy),
        "roc_approximated": float(roc_approximated),
        "f1_score": float(f1_score),
        "per_class_sensitivity": sensitivities.tolist(),
        "per_class_specificity": specificities.tolist(),
        "per_class_precision": precisions.tolist(),
        "confusion_matrix": cm.tolist(),
    }

    if class_names is not None:
        result["class_names"] = class_names

    return result


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: int = 1,
    average: str = "macro",
    class_names: Optional[dict[int, str]] = None,
) -> dict:
    """Compute classification metrics automatically detecting binary/multiclass.

    Args:
        y_true: Ground truth labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        positive_class: Label considered as positive class for binary classification
        average: Averaging method for multiclass ("macro", "micro", "weighted")
        class_names: Optional dict mapping class indices to string names
                     e.g., {0: "Benign", 1: "Malignant"}

    Returns:
        dict containing all computed metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    num_classes = max(y_true.max(), y_pred.max()) + 1

    if num_classes == 2:
        return compute_binary_metrics(y_true, y_pred, positive_class=positive_class, class_names=class_names)
    else:
        return compute_multiclass_metrics(y_true, y_pred, average=average, class_names=class_names)


def compute_metrics_from_results(
    results: dict,
    positive_class: int = 1,
    average: str = "macro",
    class_names: Optional[dict[int, str]] = None,
) -> dict:
    """Compute metrics from inference results dictionary.

    This function is designed to work with the output of
    SlideEmbeddingCalculator.compute_with_predictions().

    Args:
        results: dict with keys "labels" and "predictions"
        positive_class: Label considered as positive class for binary classification
        average: Averaging method for multiclass
        class_names: Optional dict mapping class indices to string names
                     e.g., {0: "Benign", 1: "Malignant"}

    Returns:
        dict containing all computed metrics
    """
    return compute_metrics(
        y_true=results["labels"],
        y_pred=results["predictions"],
        positive_class=positive_class,
        average=average,
        class_names=class_names,
    )


def format_metrics_table(
    metrics: dict,
    title: str = "Classification Metrics",
    precision_digits: int = 4,
) -> str:
    """Format metrics as a readable table.

    Args:
        metrics: dict of computed metrics
        title: Table title
        precision_digits: Number of decimal places

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 50)
    lines.append(f" {title}")
    lines.append("=" * 50)

    # Show class label information for binary classification
    if "positive_class" in metrics:
        class_names = metrics.get("class_names", {})
        pos_class = metrics["positive_class"]
        neg_class = metrics["negative_class"]
        pos_name = class_names.get(pos_class, str(pos_class))
        neg_name = class_names.get(neg_class, str(neg_class))

        lines.append("")
        lines.append(f"  Positive class: {pos_class} ({pos_name})")
        lines.append(f"  Negative class: {neg_class} ({neg_name})")

    # Main metrics
    metric_names = [
        ("accuracy", "Accuracy"),
        ("sensitivity", "Sensitivity (Recall)"),
        ("specificity", "Specificity"),
        ("precision", "Precision"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("roc_approximated", "ROC Approximated"),
        ("f1_score", "F1-Score"),
    ]

    max_label_len = max(len(label) for _, label in metric_names)

    lines.append("")
    for key, label in metric_names:
        if key in metrics:
            value = metrics[key]
            lines.append(f"  {label:<{max_label_len}} : {value:.{precision_digits}f}")

    # Confusion matrix counts if available (binary)
    if "tp" in metrics:
        class_names = metrics.get("class_names", {})
        pos_class = metrics["positive_class"]
        neg_class = metrics["negative_class"]
        pos_name = class_names.get(pos_class, str(pos_class))
        neg_name = class_names.get(neg_class, str(neg_class))

        lines.append("")
        lines.append("-" * 50)
        lines.append(" Confusion Matrix Counts")
        lines.append("-" * 50)
        lines.append(f"  True Positives (TP)  : {metrics['tp']:>4}  (True={pos_name}, Pred={pos_name})")
        lines.append(f"  True Negatives (TN)  : {metrics['tn']:>4}  (True={neg_name}, Pred={neg_name})")
        lines.append(f"  False Positives (FP) : {metrics['fp']:>4}  (True={neg_name}, Pred={pos_name})")
        lines.append(f"  False Negatives (FN) : {metrics['fn']:>4}  (True={pos_name}, Pred={neg_name})")

    # Per-class metrics if available (multiclass)
    if "per_class_sensitivity" in metrics:
        class_names = metrics.get("class_names", {})
        lines.append("")
        lines.append("-" * 50)
        lines.append(" Per-Class Metrics")
        lines.append("-" * 50)
        num_classes = len(metrics["per_class_sensitivity"])

        # Determine max class name length for formatting
        max_name_len = 8
        for c in range(num_classes):
            name = class_names.get(c, str(c))
            max_name_len = max(max_name_len, len(name))

        lines.append(f"  {'Class':<{max_name_len}} {'Sensitivity':<12} {'Specificity':<12} {'Precision':<12}")
        lines.append(f"  {'-'*max_name_len} {'-'*12} {'-'*12} {'-'*12}")
        for c in range(num_classes):
            name = class_names.get(c, str(c))
            sens = metrics["per_class_sensitivity"][c]
            spec = metrics["per_class_specificity"][c]
            prec = metrics["per_class_precision"][c]
            lines.append(
                f"  {name:<{max_name_len}} {sens:<12.{precision_digits}f} "
                f"{spec:<12.{precision_digits}f} {prec:<12.{precision_digits}f}"
            )

    # Confusion matrix if available (multiclass)
    if "confusion_matrix" in metrics:
        class_names = metrics.get("class_names", {})
        lines.append("")
        lines.append("-" * 50)
        lines.append(" Confusion Matrix")
        lines.append("-" * 50)
        cm = metrics["confusion_matrix"]
        num_classes = len(cm)

        # Determine column width based on class names
        col_width = 6
        for c in range(num_classes):
            name = class_names.get(c, str(c))
            col_width = max(col_width, len(name) + 1)

        # Header
        header = "  True\\Pred " + " ".join(f"{class_names.get(c, str(c)):>{col_width}}" for c in range(num_classes))
        lines.append(header)
        lines.append("  " + "-" * (11 + (col_width + 1) * num_classes))

        for i in range(num_classes):
            row_name = class_names.get(i, str(i))
            row = f"  {row_name:>10} " + " ".join(f"{cm[i][j]:>{col_width}}" for j in range(num_classes))
            lines.append(row)

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)


def print_metrics(
    metrics: dict,
    title: str = "Classification Metrics",
    precision_digits: int = 4,
) -> None:
    """Print formatted metrics table.

    Args:
        metrics: dict of computed metrics
        title: Table title
        precision_digits: Number of decimal places
    """
    print(format_metrics_table(metrics, title, precision_digits))


def metrics_to_dataframe(metrics: dict):
    """Convert metrics to pandas DataFrame for easy export.

    Args:
        metrics: dict of computed metrics

    Returns:
        pandas DataFrame with metrics
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for metrics_to_dataframe")

    main_metrics = {
        "Metric": [
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "Precision",
            "Balanced Accuracy",
            "ROC Approximated",
            "F1-Score",
        ],
        "Value": [
            metrics.get("accuracy", None),
            metrics.get("sensitivity", None),
            metrics.get("specificity", None),
            metrics.get("precision", None),
            metrics.get("balanced_accuracy", None),
            metrics.get("roc_approximated", None),
            metrics.get("f1_score", None),
        ],
    }

    return pd.DataFrame(main_metrics)


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str | Path,
    class_names: Optional[dict[int, str]] = None,
    normalize: bool = False,
    title: Optional[str] = None,
    cmap: str = "Blues",
    figsize: tuple = (8, 6),
) -> None:
    """Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix of shape (num_classes, num_classes),
            where cm[i, j] = count of samples with true label i and predicted label j.
            Use compute_confusion_matrix() to generate this.
        output_path: Path to save the plot
        class_names: Mapping from class index to display name
                     e.g., {0: "Benign", 1: "Malignant"}
        normalize: If True, normalize counts to percentages per true class (row-wise)
        title: Plot title. If None, uses default.
        cmap: Matplotlib colormap name
        figsize: Figure size
    """
    num_classes = cm.shape[0]

    if class_names is None:
        class_names = {i: str(i) for i in range(num_classes)}

    labels = [class_names.get(i, str(i)) for i in range(num_classes)]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        plot_data = np.where(row_sums > 0, cm / row_sums, 0.0)
        colorbar_label = "Proportion"
    else:
        plot_data = cm.astype(float)
        colorbar_label = "Count"

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(plot_data, interpolation="nearest", cmap=cmap, vmin=0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=11)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate each cell
    thresh = plot_data.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            value = plot_data[i, j]
            if normalize:
                text = f"{value:.2f}\n({cm[i, j]})"
            else:
                text = str(cm[i, j])
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if value > thresh else "black",
            )

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title or "Confusion Matrix", fontsize=13)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix: {output_path}")
