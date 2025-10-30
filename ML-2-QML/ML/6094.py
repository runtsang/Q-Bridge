"""Enhanced classical model for Quantum‑NAT experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depth‑wise separable convolution block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class QFCModelExtended(nn.Module):
    """Classical CNN + FC with optional depth‑wise separable conv and dropout."""

    def __init__(self, dropout_prob: float = 0.0, use_separable: bool = False) -> None:
        super().__init__()
        self.use_separable = use_separable
        if use_separable:
            self.features = nn.Sequential(
                DepthwiseSeparableConv(1, 8),
                nn.ReLU(),
                nn.MaxPool2d(2),
                DepthwiseSeparableConv(8, 16),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        features = self.dropout(features)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)


__all__ = ["QFCModelExtended"]
