"""Enhanced classical convolutional network with a richer feature extractor and MLP head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Convolution → BatchNorm → ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class QuanvolutionFilter(nn.Module):
    """Feature extractor that applies a shallow CNN to 2×2 patches."""
    def __init__(self) -> None:
        super().__init__()
        # 1 input channel (grayscale), output 4 channels per patch
        self.conv1 = ConvBlock(1, 8, kernel_size=2, stride=2, padding=0)
        self.conv2 = ConvBlock(8, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, 28, 28)
        features = self.conv1(x)  # shape: (B, 8, 14, 14)
        features = self.conv2(features)  # shape: (B, 4, 14, 14)
        return features.view(x.size(0), -1)  # flatten

class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the quanvolutional filter followed by a multi‑layer perceptron head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # 4*14*14 = 784 features
        self.mlp = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.mlp(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
