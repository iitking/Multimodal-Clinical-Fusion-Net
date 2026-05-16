"""
Generates Multimodal_Clinical_Fusion_Net.ipynb matching the exact pedagogical,
cell-by-cell structure and style of FaceGuard_CNN.ipynb.
"""

import json
from pathlib import Path


def create_faceguard_style_notebook(output_path: str = "Multimodal_Clinical_Fusion_Net.ipynb"):
    cells = []

    def code(lines):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l + "\n" for l in lines]
        })

    def md(lines):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [l + "\n" for l in lines]
        })

    # Cell 0: Imports
    code([
        "import os",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import matplotlib.image as mpimg",
        "from PIL import Image",
        "import pandas as pd",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F"
    ])

    # Cell 1: Paths
    code([
        "tabular_path = 'data/raw/tabular/patient_records.csv'",
        "images_dir = 'data/raw/images/'",
        "print('Tabular Path:', tabular_path)",
        "print('Images Dir  :', images_dir)"
    ])

    # Cell 2: Read records
    code([
        "df = pd.read_csv(tabular_path)",
        "print('Number of clinical encounters:', len(df))",
        "print('Unique patients (subject_id) :', df['subject_id'].nunique())",
        "df.head()"
    ])

    # Cell 3: Separate files
    code([
        "stable_files = df[df['outcome'] == 0]['image_path'].tolist()",
        "critical_files = df[df['outcome'] == 1]['image_path'].tolist()",
        "",
        "print('Number of stable patient images   :', len(stable_files))",
        "print('Number of critical patient images :', len(critical_files))"
    ])

    # Cell 4: Labels preview
    code([
        "stable_labels = [0] * len(stable_files)",
        "critical_labels = [1] * len(critical_files)",
        "",
        "print(stable_labels[0:5])",
        "print(critical_labels[0:5])"
    ])

    # Cell 5: Length of labels
    code([
        "print(len(stable_labels))",
        "print(len(critical_labels))"
    ])

    # Cell 6: Total labels
    code([
        "labels = df['outcome'].tolist()",
        "print('Total labels:', len(labels))"
    ])

    # Cell 7: Stable image show
    code([
        "img_stable = mpimg.imread(stable_files[0])",
        "plt.imshow(img_stable, cmap='gray')",
        "plt.title(f\"Stable Patient Scan: {df[df['outcome']==0].iloc[0]['study_id']} (Outcome: 0)\")",
        "plt.axis('off')",
        "plt.show()"
    ])

    # Cell 8: Critical image show
    code([
        "img_crit = mpimg.imread(critical_files[0])",
        "plt.imshow(img_crit, cmap='gray')",
        "plt.title(f\"Critical Patient Scan: {df[df['outcome']==1].iloc[0]['study_id']} (Outcome: 1)\")",
        "plt.axis('off')",
        "plt.show()"
    ])

    # Cell 9: Image data array loading
    code([
        "image_data = []",
        "",
        "for idx, row in df.iterrows():",
        "    img_path = row['image_path']",
        "    image = Image.open(img_path)",
        "    image = image.resize((128, 128))",
        "    image = image.convert('RGB')",
        "    image = np.array(image)",
        "    image_data.append(image)"
    ])

    # Cell 10-14: Exploration of image_data
    code(["type(image_data)"])
    code(["len(image_data)"])
    code(["image_data[0]"])
    code(["type(image_data[0])"])
    code(["image_data[0].shape"])

    # Cell 15: Tabular data extraction
    code([
        "numerical_features = [",
        "    'age', 'heart_rate', 'sbp', 'dbp', 'resp_rate',",
        "    'spo2', 'temp', 'wbc', 'creatinine', 'glucose',",
        "    'hemoglobin', 'platelets'",
        "]",
        "tabular_data = df[numerical_features].values.astype(np.float32)"
    ])

    # Cell 16-18: Exploration of tabular_data
    code(["type(tabular_data)"])
    code(["tabular_data.shape"])
    code(["tabular_data[0]"])

    # Cell 19: X and Y definitions
    code([
        "X_img = np.array(image_data)",
        "X_tab = np.array(tabular_data)",
        "Y = np.array(labels)"
    ])

    # Cell 20-22: Types
    code(["type(X_img)"])
    code(["type(X_tab)"])
    code(["type(Y)"])

    # Cell 23-24: Shapes and print Y
    code([
        "print('X_img shape:', X_img.shape)",
        "print('X_tab shape:', X_tab.shape)",
        "print('Y shape    :', Y.shape)"
    ])
    code(["print(Y)"])

    # Cell 25-26: Train test split import & execution
    code(["from sklearn.model_selection import train_test_split"])
    code([
        "X_img_train, X_img_test, X_tab_train, X_tab_test, Y_train, Y_test = train_test_split(",
        "    X_img, X_tab, Y, test_size=0.2, random_state=42, stratify=Y",
        ")"
    ])
    code([
        "print('Images  :', X_img.shape, X_img_train.shape, X_img_test.shape)",
        "print('Tabular :', X_tab.shape, X_tab_train.shape, X_tab_test.shape)"
    ])

    # Cell 28: Scaling
    code([
        "# 1. Image Scaling to [0, 1]",
        "X_img_train_scaled = X_img_train / 255.0",
        "X_img_test_scaled = X_img_test / 255.0",
        "",
        "# 2. Tabular Standard Scaling (fit on train)",
        "tab_mean = X_tab_train.mean(axis=0)",
        "tab_std = X_tab_train.std(axis=0)",
        "tab_std[tab_std < 1e-6] = 1.0",
        "",
        "X_tab_train_scaled = (X_tab_train - tab_mean) / tab_std",
        "X_tab_test_scaled = (X_tab_test - tab_mean) / tab_std"
    ])

    # Cell 29-31: Inspecting scaled data
    code(["X_img_train[0]"])
    code(["X_img_train_scaled[0]"])
    code(["X_tab_train_scaled[0]"])

    # Cell 32: Deep Neural Network Architecture
    code([
        "class MultimodalClinicalNet(nn.Module):",
        "    def __init__(self, num_tabular_features=12):",
        "        super().__init__()",
        "        # CNN Vision Stream for Chest X-Ray",
        "        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)",
        "        self.bn1 = nn.BatchNorm2d(32)",
        "        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)",
        "        self.bn2 = nn.BatchNorm2d(64)",
        "        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)",
        "        self.bn3 = nn.BatchNorm2d(128)",
        "        self.pool = nn.MaxPool2d(2, 2)",
        "        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))",
        "",
        "        # Dense Stream for Tabular EHR",
        "        self.tab_fc1 = nn.Linear(num_tabular_features, 64)",
        "        self.tab_bn1 = nn.BatchNorm1d(64)",
        "        self.tab_fc2 = nn.Linear(64, 64)",
        "        self.tab_bn2 = nn.BatchNorm1d(64)",
        "",
        "        # Multimodal Fusion & Classification Head",
        "        self.fuse_fc1 = nn.Linear(128 + 64, 128)",
        "        self.fuse_bn1 = nn.BatchNorm1d(128)",
        "        self.dropout1 = nn.Dropout(0.3)",
        "        self.fuse_fc2 = nn.Linear(128, 64)",
        "        self.dropout2 = nn.Dropout(0.2)",
        "        self.out = nn.Linear(64, 2)",
        "",
        "    def forward(self, img, tab):",
        "        # Image stream [B, 3, 128, 128]",
        "        x_img = self.pool(F.relu(self.bn1(self.conv1(img))))",
        "        x_img = self.pool(F.relu(self.bn2(self.conv2(x_img))))",
        "        x_img = self.pool(F.relu(self.bn3(self.conv3(x_img))))",
        "        x_img = self.global_pool(x_img).flatten(1)  # [B, 128]",
        "",
        "        # Tabular stream [B, 12]",
        "        x_tab = F.relu(self.tab_bn1(self.tab_fc1(tab)))",
        "        x_tab = F.relu(self.tab_bn2(self.tab_fc2(x_tab)))  # [B, 64]",
        "",
        "        # Fusion & Prediction",
        "        fused = torch.cat([x_img, x_tab], dim=-1)  # [B, 192]",
        "        h = self.dropout1(F.relu(self.fuse_bn1(self.fuse_fc1(fused))))",
        "        h = self.dropout2(F.relu(self.fuse_fc2(h)))",
        "        logits = self.out(h)",
        "        return logits"
    ])

    # Cell 33: Model summary
    code([
        "device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))",
        "model = MultimodalClinicalNet(num_tabular_features=12).to(device)",
        "",
        "total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)",
        "print('Multimodal-Clinical-Fusion-Net initialized!')",
        "print(f'Device: {device}')",
        "print(f'Total Trainable Parameters: {total_params:,}')",
        "print(model)"
    ])

    # Cell 34: Compile
    code([
        "# Compile the neural network",
        "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)",
        "criterion = nn.CrossEntropyLoss()"
    ])

    # Cell 35: DataLoaders
    code([
        "# Convert numpy arrays to PyTorch Tensors & DataLoaders",
        "t_img_train = torch.tensor(X_img_train_scaled.transpose(0, 3, 1, 2), dtype=torch.float32)",
        "t_tab_train = torch.tensor(X_tab_train_scaled, dtype=torch.float32)",
        "t_y_train = torch.tensor(Y_train, dtype=torch.long)",
        "",
        "t_img_test = torch.tensor(X_img_test_scaled.transpose(0, 3, 1, 2), dtype=torch.float32)",
        "t_tab_test = torch.tensor(X_tab_test_scaled, dtype=torch.float32)",
        "t_y_test = torch.tensor(Y_test, dtype=torch.long)",
        "",
        "from torch.utils.data import TensorDataset, DataLoader",
        "train_dataset = TensorDataset(t_img_train, t_tab_train, t_y_train)",
        "test_dataset = TensorDataset(t_img_test, t_tab_test, t_y_test)",
        "",
        "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)",
        "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)"
    ])

    # Cell 36: Training fit loop
    code([
        "# Training (model.fit)",
        "epochs = 20",
        "history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}",
        "",
        "for epoch in range(1, epochs + 1):",
        "    model.train()",
        "    total_loss, correct, total = 0.0, 0, 0",
        "    for b_img, b_tab, b_y in train_loader:",
        "        b_img, b_tab, b_y = b_img.to(device), b_tab.to(device), b_y.to(device)",
        "        optimizer.zero_grad()",
        "        logits = model(b_img, b_tab)",
        "        loss = criterion(logits, b_y)",
        "        loss.backward()",
        "        optimizer.step()",
        "",
        "        total_loss += loss.item() * len(b_y)",
        "        preds = torch.argmax(logits, dim=1)",
        "        correct += (preds == b_y).sum().item()",
        "        total += len(b_y)",
        "",
        "    train_loss = total_loss / total",
        "    train_acc = correct / total",
        "",
        "    # Evaluate on test split",
        "    model.eval()",
        "    val_loss, val_correct, val_total = 0.0, 0, 0",
        "    with torch.no_grad():",
        "        for b_img, b_tab, b_y in test_loader:",
        "            b_img, b_tab, b_y = b_img.to(device), b_tab.to(device), b_y.to(device)",
        "            logits = model(b_img, b_tab)",
        "            loss = criterion(logits, b_y)",
        "            val_loss += loss.item() * len(b_y)",
        "            preds = torch.argmax(logits, dim=1)",
        "            val_correct += (preds == b_y).sum().item()",
        "            val_total += len(b_y)",
        "",
        "    test_loss = val_loss / val_total",
        "    test_acc = val_correct / val_total",
        "",
        "    history['train_loss'].append(train_loss)",
        "    history['train_acc'].append(train_acc)",
        "    history['test_loss'].append(test_loss)",
        "    history['test_acc'].append(test_acc)",
        "",
        "    print(f'Epoch [{epoch:02d}/{epochs:02d}] - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {test_loss:.4f} - val_accuracy: {test_acc:.4f}')"
    ])

    # Cell 37: Plot curves
    code([
        "# Plot Training & Validation Loss and Accuracy Curves",
        "plt.figure(figsize=(12, 4.5))",
        "",
        "plt.subplot(1, 2, 1)",
        "plt.plot(history['train_loss'], label='Train Loss', color='#2ca02c')",
        "plt.plot(history['test_loss'], label='Validation Loss', color='#d62728', linestyle='--')",
        "plt.title('Loss Curve')",
        "plt.xlabel('Epoch')",
        "plt.ylabel('Loss')",
        "plt.legend()",
        "plt.grid(alpha=0.3)",
        "",
        "plt.subplot(1, 2, 2)",
        "plt.plot(history['train_acc'], label='Train Accuracy', color='#1f77b4')",
        "plt.plot(history['test_acc'], label='Validation Accuracy', color='#ff7f0e', linestyle='--')",
        "plt.title('Accuracy Curve')",
        "plt.xlabel('Epoch')",
        "plt.ylabel('Accuracy')",
        "plt.legend()",
        "plt.grid(alpha=0.3)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ])

    # Cell 38: Evaluate
    code([
        "model.eval()",
        "test_loss, test_correct, test_total = 0.0, 0, 0",
        "with torch.no_grad():",
        "    for b_img, b_tab, b_y in test_loader:",
        "        b_img, b_tab, b_y = b_img.to(device), b_tab.to(device), b_y.to(device)",
        "        logits = model(b_img, b_tab)",
        "        loss = criterion(logits, b_y)",
        "        test_loss += loss.item() * len(b_y)",
        "        preds = torch.argmax(logits, dim=1)",
        "        test_correct += (preds == b_y).sum().item()",
        "        test_total += len(b_y)",
        "",
        "loss = test_loss / test_total",
        "accuracy = test_correct / test_total",
        "",
        "print('Test Loss =', loss)",
        "print('Test Accuracy =', accuracy)"
    ])

    # Cell 39: Single prediction
    code([
        "from PIL import Image",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        "# Sample Chest X-ray image for testing",
        "input_image_path = df[df['outcome'] == 1].iloc[0]['image_path']",
        "print('Input Image Path:', input_image_path)",
        "",
        "input_image = Image.open(input_image_path)",
        "plt.imshow(input_image, cmap='gray')",
        "plt.title('Patient Chest X-Ray')",
        "plt.axis('off')",
        "plt.show()",
        "",
        "# Image Preprocessing",
        "input_image = input_image.convert('RGB')",
        "input_image = input_image.resize((128, 128))",
        "input_image = np.array(input_image).astype('float32') / 255.0",
        "input_img_tensor = torch.tensor(input_image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)",
        "",
        "# Tabular Preprocessing",
        "sample_patient_record = df[df['outcome'] == 1].iloc[0]",
        "raw_tabular_input = sample_patient_record[numerical_features].values.astype(np.float32)",
        "scaled_tabular_input = (raw_tabular_input - tab_mean) / tab_std",
        "input_tab_tensor = torch.tensor(scaled_tabular_input, dtype=torch.float32).unsqueeze(0).to(device)",
        "",
        "# Prediction",
        "model.eval()",
        "with torch.no_grad():",
        "    logits = model(input_img_tensor, input_tab_tensor)",
        "    input_prediction = F.softmax(logits, dim=1).cpu().numpy()[0]",
        "",
        "print('Prediction probabilities [Stable, Critical]:', input_prediction)",
        "",
        "input_pred_label = np.argmax(input_prediction)",
        "print('Predicted label:', input_pred_label)",
        "",
        "if input_pred_label == 1:",
        "    print('ALERT: The patient is at HIGH CLINICAL RISK / CRITICAL (Pulmonary Infiltrate / Sepsis detected)!')",
        "else:",
        "    print('The patient is STABLE (Normal vitals and clear lung fields).')"
    ])

    # Cell 40: Saving model and pkl
    code([
        "# PyTorch format",
        "torch.save(model.state_dict(), 'Multimodal-Clinical-Fusion-Net_model.pt')",
        "",
        "# PKL format",
        "import pickle",
        "",
        "with open('Multimodal-Clinical-Fusion-Net_model.pkl', 'wb') as file:",
        "    pickle.dump({",
        "        'model_state_dict': model.state_dict(),",
        "        'tab_mean': tab_mean,",
        "        'tab_std': tab_std,",
        "        'numerical_features': numerical_features",
        "    }, file)",
        "",
        "print('Both models saved successfully!')"
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

    print(f"FaceGuard-style Notebook generated successfully at: {output_path}")


if __name__ == "__main__":
    create_faceguard_style_notebook("Multimodal_Clinical_Fusion_Net.ipynb")
