"""Hybrid classical neural network that fuses a quanvolutional filter with a deep regression head.

The model first applies a 2×2 convolutional filter (QuanvolutionFilter) to extract
spatial features, then passes the flattened feature map through a multi‑layer
perceptron to produce a scalar regression output.  The architecture is fully
compatible with PyTorch and can be trained with standard optimizers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Simple 2×2 convolution that mimics the quanvolution filter from the original example."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        # Flatten to (batch, 4*14*14)
        return features.view(x.size(0), -1)

class EstimatorQNN(nn.Module):
    """Classical regression network that uses a quanvolution filter followed by a deep MLP."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fc1 = nn.Linear(4 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        return self.out(x)

__all__ = ["EstimatorQNN", "QuanvolutionFilter"]
