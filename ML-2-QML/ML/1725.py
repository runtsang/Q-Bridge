"""Enhanced classical quanvolution network with depth‑wise separable convolutions and residual connections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv2d(nn.Module):
    """Depth‑wise separable convolution: depth‑wise conv followed by point‑wise conv."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=0, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class QuanvolutionNet(nn.Module):
    """Classical quanvolution network with residual connections and a small MLP head."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Stage 1: 1 → 32 channels
        self.stage1 = DepthwiseSeparableConv2d(1, 32)
        # Residual path to match channel dimension
        self.residual_conv = nn.Conv2d(1, 32, kernel_size=1, bias=False)
        self.res_bn = nn.BatchNorm2d(32)

        # Stage 2: 32 → 64 channels
        self.stage2 = DepthwiseSeparableConv2d(32, 64)

        # Flatten and classifier
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        out = self.stage1(x)
        # Residual addition
        residual = self.residual_conv(x)
        residual = self.res_bn(residual)
        out = out + residual
        # Stage 2
        out = self.stage2(out)
        out = self.flatten(out)
        logits = self.classifier(out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
