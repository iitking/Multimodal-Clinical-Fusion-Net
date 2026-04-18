"""
Clinical Visualizations & Diagnostic Curves.

Generates ROC Curve, PR Curve, Confusion Matrix, and Training Trajectory plots.
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str = "outputs/plots"
) -> Dict[str, str]:
    """
    Plots Receiver Operating Characteristic and Precision-Recall Curves.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_plots = {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Multimodal Fusion Net (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Chance (AUC = 0.500)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
        plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        roc_path = str(Path(output_dir) / "roc_curve.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()
        saved_plots["roc_curve"] = roc_path

        # 2. Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        prevalence = np.mean(y_true)

        plt.figure(figsize=(7, 6))
        plt.plot(rec, prec, color="#d62728", lw=2.5, label=f"Multimodal Fusion Net (AUPRC = {auprc:.3f})")
        plt.axhline(y=prevalence, color="gray", linestyle="--", lw=1.5, label=f"Baseline Prevalence ({prevalence:.1%})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall (Sensitivity)", fontsize=12)
        plt.ylabel("Precision (PPV)", fontsize=12)
        plt.title("Precision-Recall (PR) Curve", fontsize=14, fontweight="bold")
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        pr_path = str(Path(output_dir) / "pr_curve.png")
        plt.savefig(pr_path, dpi=300)
        plt.close()
        saved_plots["pr_curve"] = pr_path

    except Exception as e:
        print(f"Skipping plot generation: {e}")

    return saved_plots


def plot_training_history(
    history: Dict[str, List[float]],
    output_dir: str = "outputs/plots"
) -> str:
    """
    Plots Loss and AUROC trajectory across epochs.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    history_path = str(Path(output_dir) / "training_curves.png")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(epochs, history["train_loss"], "o-", color="#2ca02c", label="Train Loss", lw=2)
        ax1.plot(epochs, history["val_loss"], "s--", color="#d62728", label="Val Loss", lw=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Binary Cross Entropy Loss", fontsize=12)
        ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)

        # AUROC plot
        ax2.plot(epochs, history["train_auroc"], "o-", color="#1f77b4", label="Train AUROC", lw=2)
        ax2.plot(epochs, history["val_auroc"], "s--", color="#ff7f0e", label="Val AUROC", lw=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("AUROC", fontsize=12)
        ax2.set_title("Training & Validation AUROC", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(history_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Skipping training curves plot: {e}")

    return history_path
