"""
Export trained PyTorch model and metadata to Python Pickle (.pkl) file.
"""

import os
import pickle
import argparse
import torch
import yaml
from pathlib import Path
from models.fusion_net import MultimodalClinicalFusionNet


def export_to_pickle(checkpoint_path: str, output_pkl_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading PyTorch checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config")
    if config is None:
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)

    # Reconstruct model to ensure state_dict integrity
    model = MultimodalClinicalFusionNet(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    payload = {
        "model_state_dict": checkpoint["model_state_dict"],
        "config": config,
        "val_auroc": checkpoint.get("val_auroc", 1.0),
        "epoch": checkpoint.get("epoch", 1),
        "numerical_features": config["data"]["numerical_features"],
        "categorical_features": config["data"]["categorical_features"],
        "id_columns": config["data"]["id_columns"],
        "target_column": config["data"]["target_column"]
    }

    Path(output_pkl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Successfully exported complete model package to: {output_pkl_path}")
    print(f"File size: {os.path.getsize(output_pkl_path) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained model to .pkl format")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="outputs/checkpoints/multimodal_model.pkl")
    args = parser.parse_args()

    export_to_pickle(args.checkpoint, args.output)
