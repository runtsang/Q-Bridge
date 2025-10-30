"""Classic hybrid model: CNN feature extractor + Transformer classifier.

The public API mirrors the original QuantumNAT but uses only classical
operations.  A quantum variant can be built by subclassing or by
replacing the transformer with a quantum implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Classical CNN feature extractor
# --------------------------------------------------------------------------- #
class CNNFeatureExtractor(nn.Module):
    """Convolutional backbone used to extract image features.

    The architecture matches the original QFCModel: two conv‑max‑pool
    stages, producing a 7×7 feature map that is later flattened.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, int] = (8, 16),
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# --------------------------------------------------------------------------- #
#  Classical transformer block
# --------------------------------------------------------------------------- #
class ClassicalTransformerBlock(nn.Module):
    """Single transformer encoder block using only classical layers."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridTransformer(nn.Module):
    """Stack of transformer blocks."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ClassicalTransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# --------------------------------------------------------------------------- #
#  Main hybrid model
# --------------------------------------------------------------------------- #
class QuantumHybridNAT(nn.Module):
    """CNN + Transformer text/image classifier with a classical backend.

    The class name is kept identical to the quantum variant for API
    symmetry.  Parameters can be tuned to change the depth, width, and
    number of heads.
    """

    def __init__(
        self,
        image_channels: int = 1,
        num_classes: int = 4,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cnn = CNNFeatureExtractor(image_channels)
        self.embed = nn.Linear(16 * 7 * 7, embed_dim)
        self.transformer = HybridTransformer(
            embed_dim, num_heads, num_layers, ffn_dim, dropout
        )
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        features = self.cnn(x)          # [B, 16, 7, 7]
        flat = features.view(features.size(0), -1)  # [B, 16*7*7]
        embedded = self.embed(flat).unsqueeze(1)    # [B, 1, embed_dim]
        transformed = self.transformer(embedded)     # [B, 1, embed_dim]
        pooled = transformed.mean(dim=1)            # [B, embed_dim]
        out = self.classifier(pooled)
        return out


__all__ = ["QuantumHybridNAT"]
