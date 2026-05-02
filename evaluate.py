"""
Evaluation Pipeline for Multimodal-Clinical-Fusion-Net.

Evaluates the best checkpoint on the held-out test split (grouped by subject_id).
Computes AUROC, AUPRC, Sensitivity, Specificity, F1, and renders ROC & PR curves.
"""

import os
import json
import yaml
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from data.dataset import get_multimodal_dataloaders
from models.fusion_net import MultimodalClinicalFusionNet
from utils.logger import setup_logger
from utils.metrics import compute_clinical_metrics
from utils.visualization import plot_roc_and_pr_curves


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal-Clinical-Fusion-Net on Test Set")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt",
                        help="Path to saved model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold")
    args = parser.parse_args()

    logger = setup_logger("EvalMultimodal", log_file="outputs/evaluation.log")
    logger.info("Initializing Test Set Evaluation...")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}. Run train.py first.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config")
    if config is None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} (Val AUROC: {checkpoint.get('val_auroc', 0.0):.4f})")
    logger.info(f"Using compute device: {device}")

    # Load test dataloader
    processed_path = config["data"]["processed_metadata_path"]
    df = pd.read_csv(processed_path)
    _, _, test_loader = get_multimodal_dataloaders(df, config)
    logger.info(f"Test Set contains {len(test_loader.dataset)} paired patient encounters.")

    # Reconstruct Model
    model = MultimodalClinicalFusionNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_targets = []
    all_probs = []
    all_subject_ids = []
    all_study_ids = []

    logger.info("Running inference on test set...")
    with torch.no_grad():
        for images, tabular, labels, metadata in test_loader:
            images = images.to(device)
            tabular = tabular.to(device)

            logits = model(images, tabular)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy().tolist())
            all_subject_ids.extend(metadata["subject_id"])
            all_study_ids.extend(metadata["study_id"])

    # Compute Clinical Metrics
    metrics = compute_clinical_metrics(all_targets, all_probs, threshold=args.threshold)

    # Print Summary Table
    print("\n" + "=" * 60)
    print("      MULTIMODAL CLINICAL FUSION NET - TEST REPORT")
    print("=" * 60)
    print(f"  Test Encounters Evaluated : {len(all_targets)}")
    print(f"  Decision Threshold        : {args.threshold:.2f}")
    print(f"  AUROC (ROC-AUC)           : {metrics.get('auroc', 0.0):.4f}")
    print(f"  AUPRC (Average Precision) : {metrics.get('auprc', 0.0):.4f}")
    print(f"  Sensitivity (Recall)      : {metrics.get('sensitivity', 0.0):.4f}")
    print(f"  Specificity               : {metrics.get('specificity', 0.0):.4f}")
    print(f"  Precision (PPV)           : {metrics.get('precision', 0.0):.4f}")
    print(f"  F1-Score                  : {metrics.get('f1_score', 0.0):.4f}")
    print(f"  Overall Accuracy          : {metrics.get('accuracy', 0.0):.4f}")
    print(f"  Brier Score (Calibration) : {metrics.get('brier_score', 0.0):.4f}")
    print("-" * 60)
    print(f"  Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")
    print("=" * 60 + "\n")

    # Generate & Save Plots
    plots_dir = config.get("evaluation", {}).get("plots_dir", "outputs/plots")
    saved_plots = plot_roc_and_pr_curves(np.array(all_targets), np.array(all_probs), output_dir=plots_dir)
    for plot_name, plot_path in saved_plots.items():
        logger.info(f"Saved {plot_name} to: {plot_path}")

    # Save Detailed Report
    reports_dir = Path(config.get("evaluation", {}).get("reports_dir", "outputs/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "test_evaluation_report.json"

    report_payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_val_auroc": checkpoint.get("val_auroc", 0.0),
        "test_encounters": len(all_targets),
        "decision_threshold": args.threshold,
        "metrics": metrics,
        "predictions_preview": [
            {"subject_id": sub, "study_id": std, "ground_truth": tgt, "predicted_risk": round(prob, 4)}
            for sub, std, tgt, prob in zip(all_subject_ids[:10], all_study_ids[:10], all_targets[:10], all_probs[:10])
        ]
    }

    with open(report_file, "w") as f:
        json.dump(report_payload, f, indent=2)
    logger.info(f"Detailed JSON test evaluation report saved to: {report_file}")


if __name__ == "__main__":
    main()
