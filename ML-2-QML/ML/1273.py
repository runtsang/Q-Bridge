"""Enhanced classical quanvolution model with multi‑scale patches, attention, and skip connection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleQuanvolutionFilter(nn.Module):
    """Extracts 2×2 and 4×4 patches from a 28×28 image and flattens them into a feature vector."""
    def __init__(self) -> None:
        super().__init__()
        # 2×2 patches: 14×14 patches, 4 channels
        self.conv_2x2 = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # 4×4 patches: 7×7 patches, 4 channels
        self.conv_4x4 = nn.Conv2d(1, 4, kernel_size=4, stride=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (batch, 1, 28, 28)
        feat_2x2 = self.conv_2x2(x).view(x.size(0), -1)  # 4*14*14
        feat_4x4 = self.conv_4x4(x).view(x.size(0), -1)  # 4*7*7
        return torch.cat([feat_2x2, feat_4x4], dim=1)  # 4*14*14 + 4*7*7


class AttentionModule(nn.Module):
    """Simple self‑attention that learns per‑feature weights."""
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        weights = torch.sigmoid(self.attn(x))
        return x * weights


class Quanvolution__gen221(nn.Module):
    """Multi‑scale quanvolution with attention and skip‑connection."""
    def __init__(self) -> None:
        super().__init__()
        self.filter = MultiScaleQuanvolutionFilter()
        # Dim: 4*14*14 + 4*7*7 = 784 + 196 = 980
        feature_dim = 4 * 14 * 14 + 4 * 7 * 7
        self.attention = AttentionModule(feature_dim)
        self.skip = nn.Linear(28 * 28, feature_dim)
        self.linear = nn.Linear(feature_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.filter(x)
        skip_features = self.skip(x.view(x.size(0), -1))
        combined = features + skip_features
        att_features = self.attention(combined)
        logits = self.linear(att_features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Quanvolution__gen221"]
