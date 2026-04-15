"""
Clinical Evaluation Metrics.

Computes AUROC, AUPRC, Sensitivity (Recall), Specificity, F1-Score,
Accuracy, and Confusion Matrix. Includes pure-Python fallbacks.
"""

from typing import Dict, Any, Union
import numpy as np


def compute_clinical_metrics(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes comprehensive clinical metrics.
    y_true: Ground truth binary labels (0 or 1)
    y_prob: Predicted probabilities [0.0, 1.0]
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int32)

    metrics = {}

    # Confusion Matrix Counts
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["tn"] = tn
    metrics["fn"] = fn

    total = len(y_true)
    metrics["accuracy"] = (tp + tn) / total if total > 0 else 0.0

    # Sensitivity (Recall / True Positive Rate)
    actual_positives = tp + fn
    metrics["sensitivity"] = tp / actual_positives if actual_positives > 0 else 0.0

    # Specificity (True Negative Rate)
    actual_negatives = tn + fp
    metrics["specificity"] = tn / actual_negatives if actual_negatives > 0 else 0.0

    # Precision / PPV
    pred_positives = tp + fp
    precision = tp / pred_positives if pred_positives > 0 else 0.0
    metrics["precision"] = precision

    # F1-Score
    if precision + metrics["sensitivity"] > 0:
        metrics["f1_score"] = 2 * (precision * metrics["sensitivity"]) / (precision + metrics["sensitivity"])
    else:
        metrics["f1_score"] = 0.0

    # Brier Score (Mean Squared Probability Error)
    metrics["brier_score"] = float(np.mean((y_prob - y_true) ** 2))

    # AUROC & AUPRC (using sklearn if available)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        # Check if both classes are present in y_true
        if len(np.unique(y_true)) > 1:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        else:
            metrics["auroc"] = 0.5
            metrics["auprc"] = float(np.mean(y_true))
    except ImportError:
        # Fallback rank-based Mann-Whitney U for AUROC
        pos_scores = y_prob[y_true == 1]
        neg_scores = y_prob[y_true == 0]
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            count = 0
            for p in pos_scores:
                count += np.sum(p > neg_scores) + 0.5 * np.sum(p == neg_scores)
            metrics["auroc"] = float(count / (len(pos_scores) * len(neg_scores)))
        else:
            metrics["auroc"] = 0.5
        metrics["auprc"] = float(precision)

    return metrics
