"""Enhanced classical convolutional classifier with residual blocks and depth control."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A standard residual block with two 3×3 convolutions and batch‑norm."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

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

class QuanvolutionNet(nn.Module):
    """Classical quanvolution‑inspired network with a residual backbone."""
    def __init__(self, depth: int = 3, num_classes: int = 10) -> None:
        """
        Args:
            depth: number of residual blocks per stage.
            num_classes: number of output classes.
        """
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(*[ResidualBlock(16, 16) for _ in range(depth)])
        self.stage2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2),
            *[ResidualBlock(32, 32) for _ in range(depth - 1)]
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            *[ResidualBlock(64, 64) for _ in range(depth - 1)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
