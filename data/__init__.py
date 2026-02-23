"""Data module for Multimodal Clinical Fusion Net."""
from .dataset import MultimodalClinicalDataset, get_multimodal_dataloaders, get_transforms
from .preprocess import (
    load_and_pair_data,
    split_by_patient,
    fit_and_transform_tabular,
    prepare_processed_data,
)

# Aliases for convenience
create_dataloaders = get_multimodal_dataloaders

__all__ = [
    "MultimodalClinicalDataset",
    "get_multimodal_dataloaders",
    "create_dataloaders",
    "get_transforms",
    "load_and_pair_data",
    "split_by_patient",
    "fit_and_transform_tabular",
    "prepare_processed_data",
]
