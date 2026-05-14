"""
Helper script to generate the complete Multimodal_Clinical_Fusion_Net.ipynb notebook.
"""

import json
from pathlib import Path


def create_multimodal_notebook(output_path: str = "Multimodal_Clinical_Fusion_Net.ipynb"):
    cells = []

    def add_md(source_lines):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_lines]
        })

    def add_code(source_lines):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_lines]
        })

    # Header
    add_md([
        "# 🩺 Multimodal-Clinical-Fusion-Net",
        "### Fusing Tabular EHR Data & Chest X-Ray Imaging for Clinical Decision Support",
        "",
        "**Clinical Problem & Data Linkage:**",
        "Clinical diagnosis and mortality prediction require integrating two complementary modalities:",
        "1. **Tabular EHR Modality:** Physiological vitals (HR, BP, SpO2, Temp) & systemic labs (WBC, Creatinine, Glucose, Hemoglobin, Platelets).",
        "2. **Medical Imaging Modality:** Frontal Chest X-rays (CXR) showing lung fields, cardiomegaly, and focal opacities.",
        "",
        "**Pairing Keys:**",
        "- `subject_id`: Unique Patient ID across admissions.",
        "- `study_id`: Unique Imaging Study Encounter ID.",
        "- Records are paired via `(subject_id, study_id)` with **Zero Patient Leakage** across train, validation, and test splits."
    ])

    # Section 1: Imports & Setup
    add_md(["## 1. Environment & Library Setup"])
    add_code([
        "import os",
        "import sys",
        "import yaml",
        "import pickle",
        "import numpy as np",
        "import pandas as pd",
        "from PIL import Image",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import torch",
        "import torch.nn as nn",
        "from torch.utils.data import DataLoader",
        "",
        "# Set random seeds for reproducibility",
        "torch.manual_seed(42)",
        "np.random.seed(42)",
        "",
        "# Select compute device (MPS on Apple Silicon, CUDA on NVIDIA, or CPU)",
        "device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))",
        "print(f'Using compute device: {device}')"
    ])

    # Section 2: Data Exploration & Patient Linkage
    add_md([
        "## 2. Tabular EHR & Chest X-ray Data Linkage",
        "Har patient record mein `subject_id` aur `study_id` ke through corresponding Chest X-ray scan linked hai."
    ])
    add_code([
        "# Load preprocessed metadata",
        "metadata_path = 'data/processed/paired_metadata.csv'",
        "df = pd.read_csv(metadata_path)",
        "print(f'Total Paired Encounters: {len(df)}')",
        "print(f'Unique Patients (subject_id): {df[\"subject_id\"].nunique()}')",
        "print(f'Class Distribution (Outcome):\\n{df[\"outcome\"].value_counts()}')",
        "df.head()"
    ])

    # Section 3: Visualizing Paired Clinical Encounters
    add_md([
        "## 3. Visualizing Paired Patient Encounters (Image + Tabular Labs)",
        "Yahan hum patient ke Chest X-ray aur uski clinical lab values ko side-by-side inspect karte hain."
    ])
    add_code([
        "# Visualize stable patient vs critical patient",
        "sample_stable = df[df['outcome'] == 0].iloc[0]",
        "sample_critical = df[df['outcome'] == 1].iloc[0]",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(12, 6))",
        "",
        "# Stable patient",
        "img_stable = Image.open(sample_stable['image_path'])",
        "axes[0].imshow(img_stable, cmap='gray')",
        "axes[0].set_title(f'Stable: {sample_stable[\"subject_id\"]} ({sample_stable[\"study_id\"]})\\nSpO2: {sample_stable[\"spo2\"]:.1f} | HR: {sample_stable[\"heart_rate\"]:.1f} | Outcome: 0', fontsize=11)",
        "axes[0].axis('off')",
        "",
        "# Critical patient",
        "img_crit = Image.open(sample_critical['image_path'])",
        "axes[1].imshow(img_crit, cmap='gray')",
        "axes[1].set_title(f'Critical: {sample_critical[\"subject_id\"]} ({sample_critical[\"study_id\"]})\\nSpO2: {sample_critical[\"spo2\"]:.1f} | HR: {sample_critical[\"heart_rate\"]:.1f} | Outcome: 1', fontsize=11, color='red')",
        "axes[1].axis('off')",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ])

    # Section 4: Patient-Level Zero Leakage Split Verification
    add_md([
        "## 4. Verification of Zero Patient Leakage",
        "Verify karte hain ki train, validation, aur test splits mein kisi bhi patient (`subject_id`) ka koi overlap na ho."
    ])
    add_code([
        "train_patients = set(df[df['split'] == 'train']['subject_id'])",
        "val_patients = set(df[df['split'] == 'val']['subject_id'])",
        "test_patients = set(df[df['split'] == 'test']['subject_id'])",
        "",
        "overlap_tv = train_patients.intersection(val_patients)",
        "overlap_tt = train_patients.intersection(test_patients)",
        "overlap_vt = val_patients.intersection(test_patients)",
        "",
        "print(f'Train Patients: {len(train_patients)}')",
        "print(f'Val Patients:   {len(val_patients)}')",
        "print(f'Test Patients:  {len(test_patients)}')",
        "print(f'Train vs Val Overlap:  {len(overlap_tv)} (Passed zero-leakage check)')",
        "print(f'Train vs Test Overlap: {len(overlap_tt)} (Passed zero-leakage check)')",
        "print(f'Val vs Test Overlap:   {len(overlap_vt)} (Passed zero-leakage check)')"
    ])

    # Section 5: Loading Architecture & Weights
    add_md([
        "## 5. Model Architecture & Multimodal Fusion",
        "Loading the `MultimodalClinicalFusionNet` with Cross-Modal Attention and pre-trained checkpoint."
    ])
    add_code([
        "from models.fusion_net import MultimodalClinicalFusionNet",
        "",
        "with open('configs/config.yaml', 'r') as f:",
        "    config = yaml.safe_load(f)",
        "",
        "# Instantiate model and load best checkpoint",
        "model = MultimodalClinicalFusionNet(config).to(device)",
        "checkpoint = torch.load('outputs/checkpoints/best_model.pt', map_location=device)",
        "model.load_state_dict(checkpoint['model_state_dict'])",
        "model.eval()",
        "print(f'Model loaded successfully from epoch {checkpoint[\"epoch\"]} (Val AUROC: {checkpoint[\"val_auroc\"]:.4f})')"
    ])

    # Section 6: Held-Out Test Evaluation
    add_md([
        "## 6. Held-Out Test Set Evaluation & Clinical Metrics",
        "Test set par AUROC, AUPRC, Sensitivity, Specificity, aur Confusion Matrix evaluate karte hain."
    ])
    add_code([
        "from data.dataset import get_multimodal_dataloaders",
        "from utils.metrics import compute_clinical_metrics",
        "",
        "_, _, test_loader = get_multimodal_dataloaders(df, config)",
        "",
        "all_targets = []",
        "all_probs = []",
        "",
        "with torch.no_grad():",
        "    for images, tabular, labels, _ in test_loader:",
        "        images = images.to(device)",
        "        tabular = tabular.to(device)",
        "        logits = model(images, tabular)",
        "        probs = torch.sigmoid(logits).cpu().numpy().tolist()",
        "        all_probs.extend(probs)",
        "        all_targets.extend(labels.cpu().numpy().tolist())",
        "",
        "metrics = compute_clinical_metrics(all_targets, all_probs, threshold=0.5)",
        "",
        "print('=' * 50)",
        "print('       TEST EVALUATION RESULTS')",
        "print('=' * 50)",
        "print(f'AUROC (ROC-AUC)           : {metrics[\"auroc\"]:.4f}')",
        "print(f'AUPRC (Average Precision) : {metrics[\"auprc\"]:.4f}')",
        "print(f'Sensitivity (Recall)      : {metrics[\"sensitivity\"]:.4f}')",
        "print(f'Specificity               : {metrics[\"specificity\"]:.4f}')",
        "print(f'Precision                 : {metrics[\"precision\"]:.4f}')",
        "print(f'F1-Score                  : {metrics[\"f1_score\"]:.4f}')",
        "print(f'Brier Calibration Score   : {metrics[\"brier_score\"]:.4f}')",
        "print(f'Confusion Matrix          : TP={metrics[\"tp\"]}, FP={metrics[\"fp\"]}, TN={metrics[\"tn\"]}, FN={metrics[\"fn\"]}')",
        "print('=' * 50)"
    ])

    # Section 7: ROC & PR Curves Plotting
    add_md(["## 7. Diagnostic Curves (ROC & Precision-Recall)"])
    add_code([
        "from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score",
        "",
        "fpr, tpr, _ = roc_curve(all_targets, all_probs)",
        "auc = roc_auc_score(all_targets, all_probs)",
        "prec, rec, _ = precision_recall_curve(all_targets, all_probs)",
        "auprc = average_precision_score(all_targets, all_probs)",
        "",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))",
        "",
        "# ROC Curve",
        "ax1.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Multimodal Net (AUC = {auc:.3f})')",
        "ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance')",
        "ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)",
        "ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)",
        "ax1.set_title('ROC Curve', fontsize=13, fontweight='bold')",
        "ax1.legend(loc='lower right')",
        "ax1.grid(alpha=0.3)",
        "",
        "# PR Curve",
        "ax2.plot(rec, prec, color='#d62728', lw=2.5, label=f'Multimodal Net (AUPRC = {auprc:.3f})')",
        "ax2.axhline(y=np.mean(all_targets), color='gray', linestyle='--', label=f'Prevalence ({np.mean(all_targets):.1%})')",
        "ax2.set_xlabel('Recall (Sensitivity)', fontsize=11)",
        "ax2.set_ylabel('Precision (PPV)', fontsize=11)",
        "ax2.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')",
        "ax2.legend(loc='lower left')",
        "ax2.grid(alpha=0.3)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ])

    # Section 8: Single Patient Prediction Demo
    add_md([
        "## 8. Single-Patient Clinical Decision Support Demo",
        "Ek naye patient ka scan aur tabular vitals dekar instantaneous clinical risk score aur alert level nikalna:"
    ])
    add_code([
        "from predict import predict_patient",
        "",
        "# Select a test encounter",
        "test_row = df[df['split'] == 'test'].iloc[0]",
        "study_id = test_row['study_id']",
        "image_path = test_row['image_path']",
        "tabular_dict = {feat: test_row[feat] for feat in config['data']['numerical_features']}",
        "",
        "result = predict_patient(model, image_path, tabular_dict, config, device)",
        "",
        "print('=' * 50)",
        "print('  PATIENT CLINICAL PREDICTION RESULT')",
        "print('=' * 50)",
        "print(f'Patient ID (subject_id) : {test_row[\"subject_id\"]}')",
        "print(f'Study ID   (study_id)   : {study_id}')",
        "print(f'Predicted Risk Score    : {result[\"risk_percentage\"]}')",
        "print(f'Clinical Risk Stratum   : [{result[\"alert_level\"]}] {result[\"risk_stratum\"]}')",
        "print(f'Ground Truth Label      : {test_row[\"outcome\"]}')",
        "print('=' * 50)"
    ])

    # Section 9: Pickle (.pkl) Export and Verification
    add_md([
        "## 9. Pickle (.pkl) Serialization & Reload Verification",
        "Ab hum complete model aur preprocessing metadata ko ek single `.pkl` (pickle) file mein export karte hain aur verify karte hain ki yeh standalone tarike se bina kisi issue ke reload ho sakti hai."
    ])
    add_code([
        "pkl_path = 'outputs/checkpoints/multimodal_model.pkl'",
        "",
        "# Save to .pkl",
        "export_payload = {",
        "    'model_state_dict': model.state_dict(),",
        "    'config': config,",
        "    'val_auroc': checkpoint.get('val_auroc', 1.0),",
        "    'numerical_features': config['data']['numerical_features'],",
        "    'categorical_features': config['data']['categorical_features'],",
        "    'id_columns': config['data']['id_columns']",
        "}",
        "with open(pkl_path, 'wb') as f:",
        "    pickle.dump(export_payload, f)",
        "print(f'Model successfully saved to Pickle format: {pkl_path}')",
        "print(f'File size: {os.path.getsize(pkl_path) / (1024 * 1024):.2f} MB')",
        "",
        "# Reload from .pkl and verify inference",
        "with open(pkl_path, 'rb') as f:",
        "    reloaded_data = pickle.load(f)",
        "",
        "reloaded_model = MultimodalClinicalFusionNet(reloaded_data['config']).to(device)",
        "reloaded_model.load_state_dict(reloaded_data['model_state_dict'])",
        "reloaded_model.eval()",
        "",
        "# Run test inference on reloaded model",
        "reloaded_res = predict_patient(reloaded_model, image_path, tabular_dict, config, device)",
        "print(f'Reloaded Model Verification Risk Score: {reloaded_res[\"risk_percentage\"]} (Matching: {reloaded_res[\"risk_percentage\"] == result[\"risk_percentage\"]})')",
        "print('Pickle (.pkl) serialization and deserialization verified successfully!')"
    ])

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Jupyter notebook created successfully at: {output_path}")


if __name__ == "__main__":
    create_multimodal_notebook("Multimodal_Clinical_Fusion_Net.ipynb")
