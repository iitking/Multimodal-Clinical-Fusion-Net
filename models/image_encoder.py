"""
Medical Image Vision Backbone for Chest X-Ray Analysis.

Supports standard clinical backbones (ResNet34, ResNet50, DenseNet121)
and extracts both:
  1. Global pooled embedding: [B, embed_dim]
  2. Spatial feature tokens: [B, H*W, spatial_dim] (for cross-attention fusion)
"""

import torch
import torch.nn as nn
from typing import Tuple


class SimpleClinicalCNN(nn.Module):
    """
    Lightweight, self-contained CNN backbone for environments without pre-downloaded weights.
    Outputs both global embedding and spatial feature maps.
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1), # 7x7
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv_blocks(x)  # [B, C, 7, 7]
        b, c, h, w = features.shape

        # Spatial tokens: [B, 49, C]
        spatial_tokens = features.flatten(2).transpose(1, 2)

        # Global vector: [B, C]
        pooled = self.global_pool(features).flatten(1)
        global_embed = self.proj(pooled)

        return global_embed, spatial_tokens


class ClinicalImageEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        embed_dim: int = 256
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.embed_dim = embed_dim

        # If explicitly requested custom backbone
        if self.backbone_name in ["custom", "simple_cnn", "simpleclinicalcnn"]:
            self.custom_cnn = SimpleClinicalCNN(embed_dim=embed_dim)
            self.is_custom = True
            return

        try:
            import torchvision.models as models
            if "resnet" in self.backbone_name:
                if self.backbone_name == "resnet18":
                    try:
                        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                        base = models.resnet18(weights=weights)
                    except Exception as e:
                        print(f"Notice: Loading ResNet18 without pre-downloaded weights ({e})")
                        base = models.resnet18(weights=None)
                elif self.backbone_name == "resnet50":
                    try:
                        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                        base = models.resnet50(weights=weights)
                    except Exception as e:
                        print(f"Notice: Loading ResNet50 without pre-downloaded weights ({e})")
                        base = models.resnet50(weights=None)
                else: # resnet34 default
                    try:
                        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
                        base = models.resnet34(weights=weights)
                    except Exception as e:
                        print(f"Notice: Loading ResNet34 without pre-downloaded weights ({e})")
                        base = models.resnet34(weights=None)

                # Split ResNet into feature extractor and projection
                self.feature_extractor = nn.Sequential(
                    base.conv1,
                    base.bn1,
                    base.relu,
                    base.maxpool,
                    base.layer1,
                    base.layer2,
                    base.layer3,
                    base.layer4  # Outputs [B, in_features, 7, 7]
                )
                in_features = base.fc.in_features
                self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.proj = nn.Sequential(
                    nn.Linear(in_features, embed_dim),
                    nn.LayerNorm(embed_dim)
                )
                self.spatial_proj = nn.Linear(in_features, embed_dim)
                self.is_custom = False

            elif "densenet" in self.backbone_name:
                try:
                    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
                    base = models.densenet121(weights=weights)
                except Exception as e:
                    print(f"Notice: Loading DenseNet121 without pre-downloaded weights ({e})")
                    base = models.densenet121(weights=None)

                self.feature_extractor = base.features
                in_features = base.classifier.in_features
                self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.proj = nn.Sequential(
                    nn.Linear(in_features, embed_dim),
                    nn.LayerNorm(embed_dim)
                )
                self.spatial_proj = nn.Linear(in_features, embed_dim)
                self.is_custom = False
            else:
                self.custom_cnn = SimpleClinicalCNN(embed_dim=embed_dim)
                self.is_custom = True

        except Exception as e:
            # Fallback gracefully to self-contained CNN only if torchvision fails completely
            print(f"Torchvision unavailable, using SimpleClinicalCNN: {e}")
            self.custom_cnn = SimpleClinicalCNN(embed_dim=embed_dim)
            self.is_custom = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            global_embed: [batch_size, embed_dim]
            spatial_tokens: [batch_size, num_patches, embed_dim]
        """
        if self.is_custom:
            return self.custom_cnn(x)

        # ResNet / DenseNet forward pass
        features = self.feature_extractor(x)  # [B, C, H, W]
        b, c, h, w = features.shape

        # Global representation
        pooled = self.global_pool(features).flatten(1)
        global_embed = self.proj(pooled)

        # Spatial representation tokens
        spatial_flat = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        spatial_tokens = self.spatial_proj(spatial_flat)     # [B, H*W, embed_dim]

        return global_embed, spatial_tokens

    def freeze_backbone(self, freeze: bool = True):
        """Freezes or unfreezes visual backbone weights."""
        if not self.is_custom and hasattr(self, "feature_extractor"):
            for param in self.feature_extractor.parameters():
                param.requires_grad = not freeze
