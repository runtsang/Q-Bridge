"""Classical Quanvolution implementation with enhanced feature extraction."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class Quanvolution(nn.Module):
    """
    Classical quanvolution model.

    The filter uses a 2×2 convolution followed by a shared linear projection
    that expands each patch into a higher dimensional feature vector.
    A dropout layer regularises the representation before the final linear head.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 8,
                 kernel_size: int = 2,
                 stride: int = 2,
                 projection_expansion: int = 4,
                 dropout: float = 0.1,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        # Each 2×2 patch is expanded to out_channels * projection_expansion features
        self.proj = nn.Linear(out_channels * kernel_size * kernel_size,
                              out_channels * projection_expansion)
        self.dropout = nn.Dropout(dropout)
        # Estimate feature size: 28x28 input -> 14×14 patches
        feature_dim = 14 * 14 * out_channels * projection_expansion
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv 2×2 stride 2
        conv_out = self.conv(x)                     # (N, C, H/2, W/2)
        N, C, H, W = conv_out.shape
        # Flatten each patch
        flat = conv_out.view(N, C, -1).permute(0, 2, 1)  # (N, H*W, C)
        # Project each patch
        proj = self.proj(flat)                      # (N, H*W, C*expansion)
        # Flatten all patches
        features = proj.reshape(N, -1)
        features = self.dropout(features)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Quanvolution"]
