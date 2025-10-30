"""Classical implementation of a hybrid quanvolution architecture.

This module defines QuanvolutionHybrid, which uses a standard 2×2 convolution
followed by a residual block and a linear head.  It can be used as a
drop‑in replacement for the original QuanvolutionClassifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block with two conv layers."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class QuanvolutionHybrid(nn.Module):
    """Classical hybrid model: a 2×2 convolution acts as the initial
    quanvolution filter, a residual block refines the features, and a
    linear head classifies the output.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Initial 2×2 convolution that mimics the original quanvolution filter
        self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        # Residual block on the 4×14×14 feature map
        self.res_block = ResidualBlock(4, 4, stride=1)
        # Flatten and classify
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qfilter(x)          # (B, 4, 14, 14)
        x = self.res_block(x)        # refine
        x = x.view(x.size(0), -1)    # flatten
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
