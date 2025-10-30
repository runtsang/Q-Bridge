"""Enhanced classical convolutional network with residual connections and dual‑stage filtering."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutional layers and batch normalization."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class QuanvolutionFilter(nn.Module):
    """Dual‑stage convolutional filter with a residual block."""
    def __init__(self) -> None:
        super().__init__()
        # First stage: reduce spatial resolution
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Residual block
        self.residual = ResidualBlock(4, 4)
        # Optional second stage: 1x1 conv to mix channels
        self.conv2 = nn.Conv2d(4, 4, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)          # (B, 4, 14, 14)
        x = self.residual(x)       # (B, 4, 14, 14)
        x = self.conv2(x)          # (B, 4, 14, 14)
        return x.view(x.size(0), -1)  # (B, 4*14*14)


class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the enhanced filter followed by a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
