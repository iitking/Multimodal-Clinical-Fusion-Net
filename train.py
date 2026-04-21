"""
Training Pipeline for Multimodal-Clinical-Fusion-Net.

Orchestrates data loading, model instantiation, patient-leakage-free training,
validation tracking, early stopping, and checkpoint saving.
"""

import os
import sys
import yaml
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.preprocess import prepare_processed_data
from data.dataset import get_multimodal_dataloaders
from models.fusion_net import MultimodalClinicalFusionNet
from utils.logger import setup_logger
from utils.metrics import compute_clinical_metrics
from utils.visualization import plot_training_history


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_setting: str = "auto") -> torch.device:
    if device_setting == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_setting == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple:
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for images, tabular, labels, _ in dataloader:
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images, tabular)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(all_targets)
    metrics = compute_clinical_metrics(all_targets, all_probs)
    return epoch_loss, metrics


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for images, tabular, labels, _ in dataloader:
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)

        logits = model(images, tabular)
        loss = criterion(logits, labels)

        running_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(all_targets)
    metrics = compute_clinical_metrics(all_targets, all_probs)
    return epoch_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal-Clinical-Fusion-Net")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--fusion", type=str, default=None, choices=["concat", "gated", "cross_attention"],
                        help="Override fusion strategy")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.fusion:
        config["model"]["fusion"]["strategy"] = args.fusion
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size

    logger = setup_logger("TrainMultimodal", log_file="outputs/training.log")
    logger.info("Initializing Multimodal Clinical Fusion Network Training...")

    # Set seeds & device
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(config.get("experiment", {}).get("device", "auto"))
    logger.info(f"Using compute device: {device}")

    # Prepare Preprocessed & Paired Dataset
    processed_path = config["data"]["processed_metadata_path"]
    if not os.path.exists(processed_path):
        logger.info(f"Processed metadata not found at {processed_path}. Running preprocessor...")
        processed_df = prepare_processed_data(config)
    else:
        import pandas as pd
        logger.info(f"Loading existing preprocessed metadata from {processed_path}")
        processed_df = pd.read_csv(processed_path)

    # Dataloaders
    train_loader, val_loader, test_loader = get_multimodal_dataloaders(processed_df, config)
    logger.info(f"Dataset loaded: Train={len(train_loader.dataset)} | Val={len(val_loader.dataset)} | Test={len(test_loader.dataset)}")

    # Instantiate Model
    model = MultimodalClinicalFusionNet(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Architecture: Fusion={config['model']['fusion']['strategy']} | Trainable Params={total_params:,}")

    # Loss & Class Weighting
    pos_weight_val = config["training"].get("pos_weight", 1.0)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & Scheduler
    lr = float(config["training"].get("learning_rate", 3e-4))
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    epochs = int(config["training"].get("epochs", 15))
    freeze_epochs = int(config["model"]["image_encoder"].get("freeze_backbone_epochs", 0))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Checkpoint & Output Setup
    ckpt_dir = Path(config["training"].get("checkpoint_dir", "outputs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_model.pt"

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_auroc": [],
        "val_auroc": []
    }

    best_val_auroc = -1.0
    patience = config["training"].get("early_stopping", {}).get("patience", 5)
    epochs_no_improve = 0

    logger.info("Beginning training loop...")

    for epoch in range(1, epochs + 1):
        # Freeze/Unfreeze visual backbone for warm-up
        if epoch <= freeze_epochs:
            model.image_encoder.freeze_backbone(True)
        elif epoch == freeze_epochs + 1:
            logger.info(f"Unfreezing visual backbone at epoch {epoch} for full fine-tuning.")
            model.image_encoder.freeze_backbone(False)

        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auroc"].append(train_metrics["auroc"])
        history["val_auroc"].append(val_metrics["auroc"])

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | AUROC: {train_metrics['auroc']:.4f} || "
            f"Val Loss: {val_loss:.4f} | AUROC: {val_metrics['auroc']:.4f} | F1: {val_metrics['f1_score']:.4f} | Sens: {val_metrics['sensitivity']:.4f}"
        )

        # Checkpoint Best Model
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc": best_val_auroc,
                "config": config
            }, best_ckpt_path)
            logger.info(f"  -> Saved new best model checkpoint (Val AUROC: {best_val_auroc:.4f}) to {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    # Save training trajectory curves
    plots_dir = config["evaluation"].get("plots_dir", "outputs/plots")
    plot_training_history(history, output_dir=plots_dir)
    logger.info(f"Training history curves saved to {plots_dir}/training_curves.png")
    logger.info(f"Training completed successfully. Best Validation AUROC: {best_val_auroc:.4f}")


if __name__ == "__main__":
    main()
