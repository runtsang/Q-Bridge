"""Quantized convolutional filter with a hybrid classical–quantum dual‑branch architecture."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution: depthwise conv followed by pointwise conv."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

class ChannelAttention(nn.Module):
    """Simple channel‑wise attention (Squeeze‑and‑Excitation)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ClassicalBranch(nn.Module):
    """Classical convolutional branch with depthwise separable conv and attention."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()
        self.conv = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.attn = ChannelAttention(out_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.conv(x)).view(x.size(0), -1)

class QuantumConv2d(nn.Module):
    """Classical stand‑in for the quantum kernel: a learnable linear projection of 2×2 patches."""
    def __init__(self, in_channels: int = 1, out_features: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_channels * 2 * 2, out_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # shape: b, c, h/2, w/2, 2, 2
        b, c, ph, pw, _, _ = patches.shape
        patches = patches.contiguous().view(b, c, ph * pw, 4)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(b * ph * pw, c * 4)
        out = self.proj(patches)
        out = out.view(b, ph * pw, -1)
        return out.view(b, -1)

class QuanvolutionDualFilter(nn.Module):
    """Dual‑branch filter: classical + quantum (classical stand‑in) branches."""
    def __init__(self, in_channels: int = 1, out_features: int = 4):
        super().__init__()
        self.classical = ClassicalBranch(in_channels, out_features)
        self.quantum = QuantumConv2d(in_channels, out_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_feat = self.classical(x)
        q_feat = self.quantum(x)
        return torch.cat([cls_feat, q_feat], dim=1)

class QuanvolutionDualClassifier(nn.Module):
    """Classifier built on top of the dual‑branch filter."""
    def __init__(self, num_classes: int = 10, in_channels: int = 1, out_features: int = 4):
        super().__init__()
        self.filter = QuanvolutionDualFilter(in_channels, out_features)
        self.linear = nn.Linear(2 * out_features * 14 * 14, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionDualFilter", "QuanvolutionDualClassifier"]
