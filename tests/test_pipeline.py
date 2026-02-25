"""
Unit and Integration Tests for Multimodal-Clinical-Fusion-Net.
"""

import unittest
import torch
import pandas as pd
import numpy as np

from models.tabular_encoder import TabularMLPEncoder, TabularFeatureTokenizer
from models.image_encoder import SimpleClinicalCNN
from models.fusion_net import ConcatFusion, GatedMultimodalFusion, CrossModalAttentionFusion, MultimodalClinicalFusionNet
from data.preprocess import split_by_patient
from utils.metrics import compute_clinical_metrics


class TestMultimodalPipeline(unittest.TestCase):

    def test_tabular_encoder(self):
        batch_size = 8
        num_features = 12
        embed_dim = 64
        x = torch.randn(batch_size, num_features)
        encoder = TabularMLPEncoder(input_dim=num_features, embed_dim=embed_dim)
        out = encoder(x)
        self.assertEqual(out.shape, (batch_size, embed_dim))

    def test_image_encoder_outputs(self):
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        cnn = SimpleClinicalCNN(embed_dim=128)
        global_embed, spatial_tokens = cnn(x)
        self.assertEqual(global_embed.shape, (batch_size, 128))
        self.assertEqual(spatial_tokens.shape, (batch_size, 49, 128))

    def test_fusion_strategies(self):
        batch_size = 4
        img_dim = 128
        tab_dim = 64
        fusion_dim = 128
        h_img = torch.randn(batch_size, img_dim)
        h_tab = torch.randn(batch_size, tab_dim)
        spatial = torch.randn(batch_size, 49, img_dim)

        # 1. Concat Fusion
        concat = ConcatFusion(img_dim, tab_dim, fusion_dim)
        out_concat = concat(h_img, h_tab)
        self.assertEqual(out_concat.shape, (batch_size, fusion_dim))

        # 2. Gated Fusion
        gated = GatedMultimodalFusion(img_dim, tab_dim, fusion_dim)
        out_gated = gated(h_img, h_tab)
        self.assertEqual(out_gated.shape, (batch_size, fusion_dim))

        # 3. Cross Attention Fusion
        cross_attn = CrossModalAttentionFusion(img_dim, tab_dim, fusion_dim, num_heads=2)
        out_attn = cross_attn(h_img, h_tab, spatial_tokens=spatial)
        self.assertEqual(out_attn.shape, (batch_size, fusion_dim))

    def test_end_to_end_gradient_flow(self):
        mock_config = {
            "model": {
                "tabular_encoder": {"embed_dim": 64, "hidden_dims": [64], "dropout": 0.1, "norm_type": "layer_norm"},
                "image_encoder": {"backbone": "custom", "pretrained": False, "embed_dim": 64},
                "fusion": {"strategy": "cross_attention", "fusion_dim": 64, "num_heads": 2, "dropout": 0.1},
                "classifier": {"num_classes": 1, "hidden_dim": 32, "dropout": 0.1}
            },
            "data": {
                "numerical_features": ["f1", "f2", "f3", "f4"],
                "categorical_features": []
            }
        }
        net = MultimodalClinicalFusionNet(mock_config)

        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        tabular = torch.randn(batch_size, 4)
        targets = torch.tensor([1.0, 0.0])

        logits = net(images, tabular)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(logits, targets)
        loss.backward()

        self.assertIsNotNone(net.classifier[-1].weight.grad)
        self.assertGreater(float(loss.item()), 0.0)

    def test_patient_level_split_no_leakage(self):
        # Simulate 100 encounters across 30 patients
        data = []
        for p in range(1, 31):
            for encounter in range(np.random.randint(1, 5)):
                data.append({"subject_id": f"P_{p}", "study_id": f"S_{p}_{encounter}"})
        df = pd.DataFrame(data)

        split_df = split_by_patient(df, patient_id_col="subject_id", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        train_patients = set(split_df[split_df["split"] == "train"]["subject_id"])
        val_patients = set(split_df[split_df["split"] == "val"]["subject_id"])
        test_patients = set(split_df[split_df["split"] == "test"]["subject_id"])

        # Crucial check: Zero intersection between patient sets!
        self.assertEqual(len(train_patients.intersection(val_patients)), 0)
        self.assertEqual(len(train_patients.intersection(test_patients)), 0)
        self.assertEqual(len(val_patients.intersection(test_patients)), 0)

    def test_clinical_metrics(self):
        y_true = [1, 1, 0, 0, 1]
        y_prob = [0.9, 0.8, 0.2, 0.1, 0.7]
        metrics = compute_clinical_metrics(y_true, y_prob)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["sensitivity"], 1.0)
        self.assertAlmostEqual(metrics["specificity"], 1.0)
        self.assertAlmostEqual(metrics["auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
