"""
Clinical Inference & Risk Prediction CLI for Multimodal-Clinical-Fusion-Net.

Accepts a chest X-ray image and patient tabular EHR variables,
returns the predicted clinical risk probability and risk stratification.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd

from models.fusion_net import MultimodalClinicalFusionNet


def get_inference_transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_patient(
    model: MultimodalClinicalFusionNet,
    image_path: str,
    tabular_dict: dict,
    config: dict,
    device: torch.device
) -> dict:
    """
    Runs single-patient multimodal inference.
    """
    model.eval()
    data_cfg = config["data"]
    num_features = data_cfg.get("numerical_features", [])
    cat_features = data_cfg.get("categorical_features", [])

    # Load and transform image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    image = Image.open(image_path).convert("RGB")
    transform = get_inference_transform(tuple(data_cfg.get("image_size", [224, 224])))
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Format tabular tensor
    tabular_vals = []
    for feat in num_features + cat_features:
        val = float(tabular_dict.get(feat, 0.0))
        tabular_vals.append(val)
    tabular_tensor = torch.tensor([tabular_vals], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(image_tensor, tabular_tensor)
        risk_prob = torch.sigmoid(logits).item()

    # Clinical risk stratification
    if risk_prob < 0.25:
        stratum = "LOW RISK"
        alert_level = "GREEN"
    elif risk_prob < 0.60:
        stratum = "MODERATE RISK"
        alert_level = "YELLOW"
    else:
        stratum = "HIGH RISK / CRITICAL"
        alert_level = "RED"

    return {
        "image_path": image_path,
        "predicted_risk_prob": round(risk_prob, 4),
        "risk_percentage": f"{risk_prob * 100:.1f}%",
        "risk_stratum": stratum,
        "alert_level": alert_level,
        "tabular_inputs": tabular_dict
    }


def main():
    parser = argparse.ArgumentParser(description="Multimodal Clinical Single-Patient Inference")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Path to Chest X-ray image")
    parser.add_argument("--study_id", type=str, default=None, help="Query study_id from preprocessed dataset")
    parser.add_argument("--data_file", type=str, default="data/processed/paired_metadata.csv", help="Metadata CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint not found at '{args.checkpoint}'. Run 'python train.py' first.")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config")
    if config is None:
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    model = MultimodalClinicalFusionNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # If study_id provided, look up from metadata CSV
    if args.study_id:
        if not os.path.exists(args.data_file):
            print(f"Error: Data file {args.data_file} not found.")
            sys.exit(1)
        df = pd.read_csv(args.data_file)
        row = df[df["study_id"] == args.study_id]
        if row.empty:
            print(f"Error: study_id '{args.study_id}' not found in {args.data_file}.")
            sys.exit(1)
        row = row.iloc[0]
        image_path = str(row["image_path"])
        tabular_dict = {col: row[col] for col in config["data"]["numerical_features"] if col in row}
        patient_id = row.get("subject_id", "UNKNOWN")
    elif args.image:
        image_path = args.image
        patient_id = "MANUAL_INPUT"
        # Standard baseline values for manual input test
        tabular_dict = {
            "age": 65, "heart_rate": 105, "sbp": 95, "dbp": 60,
            "resp_rate": 24, "spo2": 91.0, "temp": 38.2, "wbc": 14.5,
            "creatinine": 2.1, "glucose": 160, "hemoglobin": 10.5, "platelets": 150
        }
    else:
        # Default demo query
        df = pd.read_csv(args.data_file) if os.path.exists(args.data_file) else None
        if df is not None and not df.empty:
            sample_row = df.iloc[0]
            args.study_id = sample_row["study_id"]
            image_path = str(sample_row["image_path"])
            tabular_dict = {col: sample_row[col] for col in config["data"]["numerical_features"] if col in sample_row}
            patient_id = sample_row.get("subject_id", "UNKNOWN")
        else:
            print("Usage: python predict.py --study_id <STUDY_ID> OR --image <IMAGE_PATH>")
            sys.exit(1)

    result = predict_patient(model, image_path, tabular_dict, config, device)

    print("\n" + "=" * 55)
    print("      CLINICAL DECISION SUPPORT - PREDICTION")
    print("=" * 55)
    print(f"  Patient ID (subject_id) : {patient_id}")
    if args.study_id:
        print(f"  Study ID   (study_id)   : {args.study_id}")
    print(f"  Chest X-ray Image       : {result['image_path']}")
    print(f"  Predicted Risk Score    : {result['risk_percentage']} ({result['predicted_risk_prob']:.4f})")
    print(f"  Clinical Risk Stratum   : [{result['alert_level']}] {result['risk_stratum']}")
    print("-" * 55)
    print("  Key Tabular EHR Parameters:")
    for k, v in list(result["tabular_inputs"].items())[:6]:
        print(f"    - {k:<15}: {v}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
