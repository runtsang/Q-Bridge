"""Enhanced classical convolutional filter with residual connections and depth.

This module defines a deeper convolutional network that emulates the original
quanvolution filter but uses a residual block to preserve gradients,
and adds a second convolutional layer for better feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A simple residual block with two conv layers."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class QuanvolutionFilter(nn.Module):
    """Deep classical filter that mimics the original quanvolution structure."""
    def __init__(self) -> None:
        super().__init__()
        # First conv: 1 -> 4 channels, 2x2 kernel, stride 2
        self.initial = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.bn_initial = nn.BatchNorm2d(4)
        # Residual block to further process features
        self.res_block = ResidualBlock(4, 4, stride=1)
        # Optional pooling to reduce dimension
        self.pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        out = self.initial(x)
        out = self.bn_initial(out)
        out = F.relu(out)
        out = self.res_block(out)
        out = self.pool(out)
        # Flatten for linear head
        return out.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Classifier head that follows the enhanced quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # 4 channels * 14 * 14 = 784 features
        self.linear = nn.Linear(4 * 14 * 14, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(self.dropout(features))
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
