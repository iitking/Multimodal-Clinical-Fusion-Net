"""
PyTorch Multimodal Dataset & DataLoader for Tabular EHR + Chest X-ray.

Synchronizes:
  - image: Tensor of shape [3, 224, 224]
  - tabular: Tensor of shape [num_features]
  - label: Tensor of shape [1]
  - metadata: dict with subject_id, study_id
"""

import os
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms


class MultimodalClinicalDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str] = None,
        target_column: str = "outcome",
        image_transform=None,
        patient_id_col: str = "subject_id",
        study_id_col: str = "study_id"
    ):
        self.df = dataframe.reset_index(drop=True)
        self.num_features = numerical_features
        self.cat_features = categorical_features or []
        self.target_col = target_column
        self.transform = image_transform
        self.patient_id_col = patient_id_col
        self.study_id_col = study_id_col

        # Pre-extract tabular numpy array for efficiency
        all_feature_cols = self.num_features + self.cat_features
        self.tabular_data = self.df[all_feature_cols].values.astype("float32")

        if self.target_col in self.df.columns:
            self.labels = self.df[self.target_col].values.astype("float32")
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]

        # 1. Load Medical Image
        img_path = str(row["image_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback blank image if file is corrupt
            print(f"Warning: Corrupt or missing image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # 2. Extract Tabular Features
        tabular_tensor = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        # 3. Label
        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label_tensor = torch.tensor(-1.0, dtype=torch.float32)

        # 4. Metadata dictionary for traceability
        metadata = {
            "subject_id": str(row[self.patient_id_col]),
            "study_id": str(row[self.study_id_col]),
            "image_path": img_path
        }

        return image, tabular_tensor, label_tensor, metadata


def get_transforms(image_size=(224, 224)):
    """
    Standard ImageNet normalization and clinical augmentation transforms.
    """
    # ImageNet statistics standard for pre-trained CNNs/ViTs
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_test_transform


def get_multimodal_dataloaders(
    processed_df: pd.DataFrame,
    config: dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoaders for train, val, and test splits.
    """
    data_cfg = config["data"]
    num_features = data_cfg.get("numerical_features", [])
    cat_features = data_cfg.get("categorical_features", [])
    target_col = data_cfg.get("target_column", "outcome")
    id_cols = data_cfg.get("id_columns", {})
    patient_id_col = id_cols.get("patient_id", "subject_id")
    study_id_col = id_cols.get("study_id", "study_id")

    img_size = tuple(data_cfg.get("image_size", [224, 224]))
    batch_size = data_cfg.get("batch_size", 16)
    num_workers = data_cfg.get("num_workers", 0)

    train_transform, val_transform = get_transforms(img_size)

    # Filter splits
    train_df = processed_df[processed_df["split"] == "train"].copy()
    val_df = processed_df[processed_df["split"] == "val"].copy()
    test_df = processed_df[processed_df["split"] == "test"].copy()

    train_ds = MultimodalClinicalDataset(
        train_df, num_features, cat_features, target_col, train_transform, patient_id_col, study_id_col
    )
    val_ds = MultimodalClinicalDataset(
        val_df, num_features, cat_features, target_col, val_transform, patient_id_col, study_id_col
    )
    test_ds = MultimodalClinicalDataset(
        test_df, num_features, cat_features, target_col, val_transform, patient_id_col, study_id_col
    )

    pin_mem = torch.cuda.is_available() and num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )

    return train_loader, val_loader, test_loader
