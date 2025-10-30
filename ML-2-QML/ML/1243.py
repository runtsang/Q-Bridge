# ML code
"""Enhanced classical quanvolution classifier with residual connections and depthwise separable convolutions."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    'Depthwise separable convolution: depthwise conv followed by pointwise conv.'
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=0, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)

class QuanvolutionFilter(nn.Module):
    'Hybrid classical quanvolution filter using depthwise separable conv and residual shortcut.'
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride)
        # Residual projection to match dimensions
        self.res_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ds_conv(x)
        res = self.res_proj(x)
        out = out + res
        out = self.bn(out)
        return F.relu(out).view(x.size(0), -1)  # flatten for linear head

class QuanvolutionClassifier(nn.Module):
    'Classical classifier with residual blocks after quanvolution filter.'
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels=in_channels)
        # After flatten, number of features = 4 * 14 * 14
        self.fc1 = nn.Linear(4 * 14 * 14, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        x = F.relu(self.bn1(self.fc1(features)))
        logits = self.fc2(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ['DepthwiseSeparableConv', 'QuanvolutionFilter', 'QuanvolutionClassifier']
