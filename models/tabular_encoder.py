"""
Tabular EHR Feature Encoder for Clinical Data.

Encodes demographic, vital sign, and laboratory measurements using
a deep residual MLP with Layer Normalization and GELU non-linearities.
"""

import torch
import torch.nn as nn
from typing import List


class TabularMLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        norm_type: str = "layer_norm"
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.input_dim = input_dim
        self.embed_dim = embed_dim

        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if norm_type == "batch_norm":
                layers.append(nn.BatchNorm1d(h_dim))
            else:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim

        # Final projection to target embedding dimension
        layers.append(nn.Linear(current_dim, embed_dim))
        layers.append(nn.LayerNorm(embed_dim))

        self.network = nn.Sequential(*layers)

        # Skip connection if dimensions align
        if input_dim == embed_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tabular tensor of shape [batch_size, input_dim]
        Returns:
            Embedding tensor of shape [batch_size, embed_dim]
        """
        out = self.network(x) + self.residual(x)
        return out


class TabularFeatureTokenizer(nn.Module):
    """
    Transforms each continuous tabular feature into an individual token vector,
    enabling token-level cross-attention with spatial image patches.
    """
    def __init__(self, num_features: int, token_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        self.token_dim = token_dim
        # Individual linear weight and bias for each continuous feature
        self.weight = nn.Parameter(torch.randn(num_features, token_dim))
        self.bias = nn.Parameter(torch.zeros(num_features, token_dim))
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, num_features]
        Returns:
            Tokens of shape [batch_size, num_features, token_dim]
        """
        # x is [B, N], expand to [B, N, 1]
        x_exp = x.unsqueeze(-1)
        # weight is [N, D], broadcasted multiplication
        tokens = x_exp * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return self.norm(tokens)
