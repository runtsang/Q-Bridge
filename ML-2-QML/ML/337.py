"""Extended quanvolution model with multi‑scale convolution and channel attention.

The model replaces the single 2×2 convolution with a set of convolutions of
different kernel sizes (2, 3, 4) to capture features at multiple scales.
A squeeze‑and‑excitation block applies channel‑wise attention, and the
resulting feature map is flattened and fed to a linear classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Squeeze‑and‑excitation channel‑attention block."""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class MultiScaleConv(nn.Module):
    """Convolution with multiple kernel sizes and stride‑2 down‑sampling."""
    def __init__(self, in_channels: int, out_channels: int, scales: list[int] = [2, 3, 4]):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in scales:
            padding = k // 2
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=padding, stride=2)
            )
        self.bn = nn.BatchNorm2d(out_channels * len(scales))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [conv(x) for conv in self.convs]
        x = torch.cat(features, dim=1)
        x = self.bn(x)
        return self.relu(x)

class QuanvolutionModel(nn.Module):
    """Hybrid classical quanvolution model with attention and multi‑scale features."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 scales: list[int] = [2, 3, 4], num_classes: int = 10):
        super().__init__()
        self.multi_conv = MultiScaleConv(in_channels, out_channels, scales)
        self.attn = ChannelAttention(out_channels * len(scales))
        self.flatten = nn.Flatten()
        # Feature map size after stride‑2 down‑sampling: 28 → 14
        self.feature_dim = out_channels * len(scales) * 14 * 14
        self.linear = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.multi_conv(x)
        x = self.attn(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionModel"]
