"""Enhanced classical convolutional filter inspired by the original quanvolution example."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseQuantumFilter(nn.Module):
    """Depth‑wise separable convolution that mimics a quantum kernel on 4×4 patches."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 4, stride: int = 2):
        super().__init__()
        # depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, groups=in_channels, bias=False)
        # pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # MLP to emulate nonlinear quantum feature map
        patch_dim = ((28 - kernel_size) // stride + 1) ** 2
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * patch_dim, out_channels * 4),
            nn.ReLU(),
            nn.Linear(out_channels * 4, out_channels * 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.depthwise(x)
        x = self.pointwise(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

class QuanvolutionPlusClassifier(nn.Module):
    """Hybrid classical network using DepthwiseQuantumFilter followed by a linear head."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = DepthwiseQuantumFilter()
        dummy = torch.zeros(1, 1, 28, 28)
        feat = self.qfilter(dummy)
        self.linear = nn.Linear(feat.shape[1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["DepthwiseQuantumFilter", "QuanvolutionPlusClassifier"]
