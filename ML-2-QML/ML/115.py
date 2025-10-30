"""Enhanced classical convolutional architecture inspired by Quanvolution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ConvBlock(nn.Module):
    """Three‑layer convolutional block followed by a residual connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.residual(out)
        return out


class EnhancedQuanvolutionFilter(nn.Module):
    """Classical feature extractor that mimics the original quanvolution filter."""

    def __init__(self) -> None:
        super().__init__()
        self.block = ConvBlock(1, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.block(x)
        return features.view(x.size(0), -1)


class EnhancedQuanvolutionClassifier(nn.Module):
    """Deeper classifier built on the enhanced filter."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = EnhancedQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["EnhancedQuanvolutionFilter", "EnhancedQuanvolutionClassifier"]
