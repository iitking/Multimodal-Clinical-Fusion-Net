"""
Multimodal Fusion Architectures & Master Clinical Network.

Implements three fusion paradigms:
  1. ConcatFusion: Direct late concatenation with non-linear projection.
  2. GatedMultimodalFusion: Dynamic Gated Multimodal Unit (GMU) weighting.
  3. CrossModalAttentionFusion: Cross-attention between Tabular tokens and Imaging spatial features.

Master Class:
  - MultimodalClinicalFusionNet
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .tabular_encoder import TabularMLPEncoder
from .image_encoder import ClinicalImageEncoder


class ConcatFusion(nn.Module):
    """Simple and effective late feature concatenation."""
    def __init__(self, img_dim: int, tab_dim: int, fusion_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(img_dim + tab_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, img_embed: torch.Tensor, tab_embed: torch.Tensor, **kwargs) -> torch.Tensor:
        combined = torch.cat([img_embed, tab_embed], dim=-1)
        return self.fusion_layer(combined)


class GatedMultimodalFusion(nn.Module):
    """
    Gated Multimodal Unit (GMU):
    Learns an input-dependent gating vector z in [0, 1] determining
    how much weight to assign to imaging vs tabular EHR.
    """
    def __init__(self, img_dim: int, tab_dim: int, fusion_dim: int, dropout: float = 0.2):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Tanh()
        )
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Tanh()
        )
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(img_dim + tab_dim, fusion_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(fusion_dim)

    def forward(self, img_embed: torch.Tensor, tab_embed: torch.Tensor, **kwargs) -> torch.Tensor:
        h_img = self.img_proj(img_embed)
        h_tab = self.tab_proj(tab_embed)

        concat_raw = torch.cat([img_embed, tab_embed], dim=-1)
        z = self.gate(concat_raw)  # [B, fusion_dim] in [0, 1]

        # Element-wise gated blend
        fused = z * h_img + (1.0 - z) * h_tab
        fused = self.out_norm(self.dropout(fused))
        return fused


class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Attention:
    1. Tabular query attends to visual spatial tokens (e.g. abnormal labs attend to lung opacity).
    2. Visual pooled query attends to tabular embedding.
    """
    def __init__(self, img_dim: int, tab_dim: int, fusion_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, fusion_dim)
        self.tab_proj = nn.Linear(tab_dim, fusion_dim)

        # Multi-Head Attention: Query=Tabular, Key/Value=Visual Spatial Tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        # Feed-forward refinement
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

    def forward(
        self,
        img_embed: torch.Tensor,
        tab_embed: torch.Tensor,
        spatial_tokens: torch.Tensor = None
    ) -> torch.Tensor:
        """
        img_embed: [B, img_dim]
        tab_embed: [B, tab_dim]
        spatial_tokens: [B, N_patches, img_dim]
        """
        h_img = self.img_proj(img_embed)  # [B, fusion_dim]
        h_tab = self.tab_proj(tab_embed)  # [B, fusion_dim]

        if spatial_tokens is not None:
            # Map spatial tokens to fusion_dim
            spatial_proj = self.img_proj(spatial_tokens)  # [B, N, fusion_dim]
            # Tabular queries visual tokens
            q = h_tab.unsqueeze(1)  # [B, 1, fusion_dim]
            k = spatial_proj        # [B, N, fusion_dim]
            v = spatial_proj

            attn_out, attn_weights = self.cross_attn(query=q, key=k, value=v)
            attn_tab = self.norm1(h_tab + attn_out.squeeze(1))
        else:
            attn_tab = h_tab

        # Concatenate attended tabular representation with global visual representation
        joint = torch.cat([h_img, attn_tab], dim=-1)
        fused = self.ffn(joint)
        return fused


class MultimodalClinicalFusionNet(nn.Module):
    """
    End-to-End Multimodal Clinical Network.
    Integrates Tabular EHR + Chest X-ray Imaging for Clinical Risk Prediction.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        data_cfg = config["data"]

        num_numerical = len(data_cfg.get("numerical_features", []))
        num_categorical = len(data_cfg.get("categorical_features", []))
        total_tabular_features = num_numerical + num_categorical

        # 1. Tabular Encoder
        tab_cfg = model_cfg.get("tabular_encoder", {})
        tab_embed_dim = tab_cfg.get("embed_dim", 128)
        self.tabular_encoder = TabularMLPEncoder(
            input_dim=total_tabular_features,
            embed_dim=tab_embed_dim,
            hidden_dims=tab_cfg.get("hidden_dims", [128, 128]),
            dropout=tab_cfg.get("dropout", 0.2),
            norm_type=tab_cfg.get("norm_type", "layer_norm")
        )

        # 2. Medical Image Encoder
        img_cfg = model_cfg.get("image_encoder", {})
        img_embed_dim = img_cfg.get("embed_dim", 256)
        self.image_encoder = ClinicalImageEncoder(
            backbone_name=img_cfg.get("backbone", "resnet34"),
            pretrained=img_cfg.get("pretrained", True),
            embed_dim=img_embed_dim
        )

        # 3. Multimodal Fusion Module
        fusion_cfg = model_cfg.get("fusion", {})
        self.fusion_strategy = fusion_cfg.get("strategy", "cross_attention").lower()
        fusion_dim = fusion_cfg.get("fusion_dim", 256)
        dropout = fusion_cfg.get("dropout", 0.2)

        if self.fusion_strategy == "concat":
            self.fusion_module = ConcatFusion(img_embed_dim, tab_embed_dim, fusion_dim, dropout)
        elif self.fusion_strategy == "gated":
            self.fusion_module = GatedMultimodalFusion(img_embed_dim, tab_embed_dim, fusion_dim, dropout)
        elif self.fusion_strategy == "cross_attention":
            self.fusion_module = CrossModalAttentionFusion(
                img_embed_dim, tab_embed_dim, fusion_dim,
                num_heads=fusion_cfg.get("num_heads", 4),
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # 4. Clinical Task Classifier Head
        clf_cfg = model_cfg.get("classifier", {})
        num_classes = clf_cfg.get("num_classes", 1)
        hidden_dim = clf_cfg.get("hidden_dim", 128)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(clf_cfg.get("dropout", 0.2)),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multimodal pipeline.
        images: [B, 3, H, W]
        tabular: [B, num_features]
        Returns: Logits of shape [B, num_classes] (or [B] if num_classes == 1)
        """
        # Encode Tabular Modality
        tab_embed = self.tabular_encoder(tabular)  # [B, tab_embed_dim]

        # Encode Medical Image Modality
        img_embed, spatial_tokens = self.image_encoder(images)

        # Multimodal Fusion
        fused = self.fusion_module(
            img_embed=img_embed,
            tab_embed=tab_embed,
            spatial_tokens=spatial_tokens
        )

        # Clinical Classification
        logits = self.classifier(fused)
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
        """
        Intelligent state_dict loader:
        1. Remaps legacy/notebook key variations (fusion. -> fusion_module., global_proj -> proj, fc1 -> network).
        2. Automatically adapts image_encoder between ResNet and custom CNN.
        """
        remapped = {}
        for k, v in state_dict.items():
            new_k = k
            if new_k.startswith("fusion."):
                new_k = new_k.replace("fusion.", "fusion_module.", 1)
            if "image_encoder.global_proj." in new_k:
                new_k = new_k.replace("image_encoder.global_proj.", "image_encoder.proj.")
            if "tabular_encoder.fc1." in new_k:
                new_k = new_k.replace("tabular_encoder.fc1.", "tabular_encoder.network.0.")
            if "tabular_encoder.norm1." in new_k:
                new_k = new_k.replace("tabular_encoder.norm1.", "tabular_encoder.network.1.")
            if "tabular_encoder.fc2." in new_k:
                new_k = new_k.replace("tabular_encoder.fc2.", "tabular_encoder.network.4.")
            if "tabular_encoder.norm2." in new_k:
                new_k = new_k.replace("tabular_encoder.norm2.", "tabular_encoder.network.5.")
            if "tabular_encoder.shortcut." in new_k:
                new_k = new_k.replace("tabular_encoder.shortcut.", "tabular_encoder.residual.")
            remapped[new_k] = v

        has_custom = any(k.startswith("image_encoder.custom_cnn.") for k in remapped.keys())
        has_feature_extractor = any(k.startswith("image_encoder.feature_extractor.") for k in remapped.keys())

        if has_custom and not getattr(self.image_encoder, "is_custom", False):
            from .image_encoder import ClinicalImageEncoder
            embed_dim = self.image_encoder.embed_dim
            dev = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
            self.image_encoder = ClinicalImageEncoder(backbone_name="custom", pretrained=False, embed_dim=embed_dim).to(dev)

        elif has_feature_extractor and getattr(self.image_encoder, "is_custom", False):
            from .image_encoder import ClinicalImageEncoder
            embed_dim = self.image_encoder.embed_dim
            dev = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
            self.image_encoder = ClinicalImageEncoder(backbone_name="resnet34", pretrained=False, embed_dim=embed_dim).to(dev)

        return super().load_state_dict(remapped, strict=strict, assign=assign)

    @torch.no_grad()
    def predict_risk(self, image: torch.Tensor, tabular: torch.Tensor) -> float:
        """Convenience method for single patient risk scoring."""
        self.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if tabular.dim() == 1:
            tabular = tabular.unsqueeze(0)

        logits = self.forward(image, tabular)
        prob = torch.sigmoid(logits).item()
        return prob
