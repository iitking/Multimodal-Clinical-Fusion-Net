"""
Data Preprocessing & Linkage Pipeline for Multimodal-Clinical-Fusion-Net.

Links Tabular EHR (demographics, vitals, labs) with Medical Imaging metadata
using `subject_id` (Patient ID) and `study_id` (Encounter ID).
Applies patient-stratified split to prevent data leakage.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


def load_and_pair_data(tabular_path: str, images_dir: str, id_cols: dict) -> pd.DataFrame:
    """
    Loads tabular data and validates linkage with medical image files.
    """
    if not os.path.exists(tabular_path):
        raise FileNotFoundError(f"Tabular data file not found: {tabular_path}")

    df = pd.read_csv(tabular_path)
    patient_id_col = id_cols.get("patient_id", "subject_id")
    study_id_col = id_cols.get("study_id", "study_id")

    # Validate required linkage keys
    for col in [patient_id_col, study_id_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required identifier column: '{col}' in tabular CSV.")

    # Validate / build image file path
    if "image_path" not in df.columns:
        # Check standard file naming: {study_id}.png or {study_id}.jpg
        def find_image(study_id):
            for ext in [".png", ".jpg", ".jpeg", ".dcm"]:
                candidate = Path(images_dir) / f"{study_id}{ext}"
                if candidate.exists():
                    return str(candidate)
            return None

        df["image_path"] = df[study_id_col].apply(find_image)
    else:
        # Ensure path exists or is relative to project root
        def resolve_path(p):
            if os.path.exists(p):
                return p
            candidate = Path(images_dir) / Path(p).name
            if candidate.exists():
                return str(candidate)
            return p

        df["image_path"] = df["image_path"].apply(resolve_path)

    # Filter out records where image does not exist
    initial_len = len(df)
    valid_mask = df["image_path"].apply(lambda p: os.path.exists(str(p)) if pd.notnull(p) else False)
    df = df[valid_mask].copy()
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"Filtered out {dropped} records with missing image files.")

    print(f"Paired {len(df)} records across {df[patient_id_col].nunique()} unique patients.")
    return df


def split_by_patient(
    df: pd.DataFrame,
    patient_id_col: str = "subject_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Splits the dataset at the PATIENT level (`subject_id`) so that
    no patient's studies appear in more than one split (no leakage).
    """
    df = df.copy()
    df["split"] = "train"

    groups = df[patient_id_col].values

    # Step 1: Split into Train + Val vs Test
    test_size = test_ratio
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

    # Step 2: Split Train + Val into Train vs Val
    val_relative_size = val_ratio / (train_ratio + val_ratio)
    train_val_df = df.iloc[train_val_idx]
    train_val_groups = train_val_df[patient_id_col].values

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=random_state)
    sub_train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_groups))

    actual_train_idx = train_val_idx[sub_train_idx]
    actual_val_idx = train_val_idx[val_idx]

    df.iloc[actual_train_idx, df.columns.get_loc("split")] = "train"
    df.iloc[actual_val_idx, df.columns.get_loc("split")] = "val"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    print("Patient-level Split Distribution:")
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        print(f"  {split_name.upper():<6}: {len(sub):>4} encounters | {sub[patient_id_col].nunique():>4} unique patients")

    return df


def fit_and_transform_tabular(
    df: pd.DataFrame,
    numerical_features: list,
    categorical_features: list
) -> tuple:
    """
    Fits standard scaling & imputation on train split only, then transforms all splits.
    Returns: (processed_df, feature_stats_dict)
    """
    df = df.copy()
    train_mask = df["split"] == "train"

    stats = {
        "num_means": {},
        "num_stds": {},
        "cat_mappings": {}
    }

    # 1. Numerical Features: Mean/Std standardization + Median imputation
    for col in numerical_features:
        if col in df.columns:
            train_vals = pd.to_numeric(df.loc[train_mask, col], errors="coerce")
            mean_val = float(train_vals.mean())
            std_val = float(train_vals.std()) if train_vals.std() > 1e-6 else 1.0

            stats["num_means"][col] = mean_val
            stats["num_stds"][col] = std_val

            # Fill missing with mean and standardize
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(mean_val)
            df[col] = (df[col] - mean_val) / std_val

    # 2. Categorical Features: Label encoding
    for col in categorical_features:
        if col in df.columns:
            unique_cats = sorted(df.loc[train_mask, col].dropna().unique())
            cat_map = {cat: idx for idx, cat in enumerate(unique_cats)}
            stats["cat_mappings"][col] = cat_map

            # Unseen or missing mapped to 0 or len(cat_map)
            df[col] = df[col].map(lambda x: cat_map.get(x, 0)).astype(int)

    return df, stats


def prepare_processed_data(config: dict) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline called before training.
    """
    data_cfg = config["data"]
    tabular_path = data_cfg["tabular_path"]
    images_dir = data_cfg["images_dir"]
    id_cols = data_cfg["id_columns"]
    patient_id_col = id_cols.get("patient_id", "subject_id")

    # 1. Load & Pair
    df = load_and_pair_data(tabular_path, images_dir, id_cols)

    # 2. Patient-level split
    df = split_by_patient(
        df,
        patient_id_col=patient_id_col,
        train_ratio=data_cfg.get("train_split", 0.70),
        val_ratio=data_cfg.get("val_split", 0.15),
        test_ratio=data_cfg.get("test_split", 0.15),
        random_state=config.get("experiment", {}).get("seed", 42)
    )

    # 3. Standardize Tabular Features (leak-free)
    num_features = data_cfg.get("numerical_features", [])
    cat_features = data_cfg.get("categorical_features", [])
    df, stats = fit_and_transform_tabular(df, num_features, cat_features)

    # 4. Save processed metadata
    output_path = data_cfg.get("processed_metadata_path", "data/processed/paired_metadata.csv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed paired metadata saved to: {output_path}")

    return df


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Multimodal Clinical Data")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    prepare_processed_data(cfg)

