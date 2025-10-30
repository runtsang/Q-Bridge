"""Hybrid classical model with a residual block and a linear classifier.

The model extends the original quanvolution filter by adding a
classical residual branch that captures low‑level spatial patterns
before the final linear head.  This enables richer feature extraction
while keeping the architecture fully classical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    """1x1 → 2x2 convolutional residual block."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class QuanvolutionGen104(nn.Module):
    """Classical hybrid classifier that first extracts residual features
    and then predicts class logits via a linear head."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Residual branch
        self.res_block = ResidualConvBlock(1, 4)
        # Flatten and linear layer
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape (batch, 1, 28, 28)
        features = self.res_block(x)          # shape (batch, 4, 14, 14)
        features = features.view(x.size(0), -1)  # flatten
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen104"]
