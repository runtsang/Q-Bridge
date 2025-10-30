"""Hybrid classical layer combining a fully connected map and a 2×2 convolutional filter."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable


class HybridQuantumLayer(nn.Module):
    """A hybrid classical layer that first applies a 1×1 fully connected map,
    then a 2×2 convolutional filter, and finally concatenates the flattened
    outputs. This design fuses the spirit of the original FCL and Quanvolution
    examples while providing a single entry point for downstream models."""

    def __init__(self, in_features: int = 1, conv_out_channels: int = 4):
        super().__init__()
        # Fully connected part (mimicking the parameterized linear in FCL)
        self.fc = nn.Linear(in_features, 1)
        # Convolution part (mimicking the 2×2 filter in Quanvolution)
        self.conv = nn.Conv2d(1, conv_out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume x shape: (batch, 1, H, W)
        # Fully connected branch
        fc_out = torch.tanh(self.fc(x.view(x.size(0), -1))).view(x.size(0), 1, 1, 1)
        # Convolution branch
        conv_out = self.conv(x)
        # Flatten and concatenate
        conv_flat = conv_out.view(x.size(0), -1)
        fc_flat = fc_out.view(x.size(0), -1)
        return torch.cat([fc_flat, conv_flat], dim=1)


__all__ = ["HybridQuantumLayer"]
