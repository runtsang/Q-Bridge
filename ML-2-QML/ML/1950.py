"""Enhanced classical convolutional architecture with depthwise separable convolution and transformer encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

class QuanvolutionFilter(nn.Module):
    """Classical depthwise separable convolution filter."""
    def __init__(self):
        super().__init__()
        self.conv = DepthwiseSeparableConv2d(1, 4, kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)  # shape: [B, 4, 14, 14]
        return features.view(x.size(0), -1)  # flatten

class QuanvolutionClassifier(nn.Module):
    """Hybrid architecture: quanvolution filter + transformer encoder + linear head."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        encoder_layer = TransformerEncoderLayer(
            d_model=4 * 14 * 14, nhead=4, dim_feedforward=512, dropout=0.1
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # [B, 4*14*14]
        seq = features.unsqueeze(0)  # [1, B, D]
        transformed = self.transformer(seq).squeeze(0)  # [B, D]
        logits = self.linear(transformed)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
