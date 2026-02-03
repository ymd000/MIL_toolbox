"""
Training log visualization script.
Plots train loss and train accuracy from outputs/fold_X/logs/version_X/metrics.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(log_dir: Path, version: str | None = None) -> pd.DataFrame:
    """Load metrics.csv from the specified log directory.

    Args:
        log_dir: Path to logs directory (e.g., outputs/fold_0/logs)
        version: Specific version to load (e.g., 'version_3'). If None, loads the latest.

    Returns:
        DataFrame with metrics data
    """
    if version is None:
        # Find the latest version
        versions = sorted(
            [d for d in log_dir.glob("version_*") if d.is_dir()],
            key=lambda x: int(x.name.split("_")[1]),
        )
        if not versions:
            raise FileNotFoundError(f"No version directories found in {log_dir}")
        version_dir = versions[-1]
    else:
        version_dir = log_dir / version

    metrics_path = version_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found at {metrics_path}")

    return pd.read_csv(metrics_path)


def plot_training_metrics(
    outputs_dir: Path,
    folds: list[int] | None = None,
    version: str | None = None,
    plot_step: bool = False,
    save_path: Path | None = None,
):
    """Plot training loss and accuracy.

    Args:
        outputs_dir: Path to outputs directory
        folds: List of fold indices to plot. If None, plots all available folds.
        version: Specific version to load. If None, loads the latest version.
        plot_step: If True, plot step-level metrics. If False, plot epoch-level metrics.
        save_path: Path to save the figure. If None, displays the plot.
    """
    # Find available folds
    if folds is None:
        fold_dirs = sorted(
            [d for d in outputs_dir.glob("fold_*") if d.is_dir()],
            key=lambda x: int(x.name.split("_")[1]),
        )
        folds = [int(d.name.split("_")[1]) for d in fold_dirs]

    if not folds:
        raise ValueError("No fold directories found")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10.colors

    for i, fold_idx in enumerate(folds):
        fold_dir = outputs_dir / f"fold_{fold_idx}"
        log_dir = fold_dir / "logs"

        if not log_dir.exists():
            print(f"Warning: logs directory not found for fold_{fold_idx}")
            continue

        try:
            df = load_metrics(log_dir, version)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        color = colors[i % len(colors)]

        if plot_step:
            # Plot step-level metrics
            step_data = df[df["train_loss_step"].notna()].copy()
            step_data["global_step"] = range(len(step_data))

            # Loss
            axes[0].plot(
                step_data["global_step"],
                step_data["train_loss_step"],
                label=f"Fold {fold_idx}",
                color=color,
                alpha=0.7,
            )

            # Accuracy
            axes[1].plot(
                step_data["global_step"],
                step_data["train_acc_step"],
                label=f"Fold {fold_idx}",
                color=color,
                alpha=0.7,
            )

            axes[0].set_xlabel("Step")
            axes[1].set_xlabel("Step")
        else:
            # Plot epoch-level metrics
            epoch_data = df[df["train_loss_epoch"].notna()].copy()

            # Loss
            axes[0].plot(
                epoch_data["epoch"],
                epoch_data["train_loss_epoch"],
                label=f"Fold {fold_idx}",
                color=color,
                marker="o",
                markersize=4,
            )

            # Accuracy
            acc_data = df[df["train_acc_epoch"].notna()].copy()
            axes[1].plot(
                acc_data["epoch"],
                acc_data["train_acc_epoch"],
                label=f"Fold {fold_idx}",
                color=color,
                marker="o",
                markersize=4,
            )

            axes[0].set_xlabel("Epoch")
            axes[1].set_xlabel("Epoch")

    # Configure loss plot
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Configure accuracy plot
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Train Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training loss and accuracy from logs")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to outputs directory (default: outputs)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Fold indices to plot (default: all folds)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Specific version to load (e.g., version_3). Default: latest version",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Plot step-level metrics instead of epoch-level",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Path to save the figure (e.g., training_plot.png)",
    )

    args = parser.parse_args()

    plot_training_metrics(
        outputs_dir=args.outputs_dir,
        folds=args.folds,
        version=args.version,
        plot_step=args.step,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
