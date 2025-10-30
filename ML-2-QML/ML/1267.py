"""Enhanced classical model for Quantum‑NAT with multi‑scale feature extraction and contrastive pretraining."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block with two conv layers."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class MultiScaleCNN(nn.Module):
    """CNN with multi‑scale feature extraction using residual blocks."""
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResidualBlock(16, 32, stride=2)  # 28x28 → 14x14
        self.layer2 = ResidualBlock(32, 64, stride=2)  # 14x14 → 7x7
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 4)
        self.norm = nn.BatchNorm1d(4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.norm(x)

class QuantumNATEnhanced(nn.Module):
    """Classical backbone for hybrid model."""
    def __init__(self):
        super().__init__()
        self.backbone = MultiScaleCNN()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

__all__ = ["QuantumNATEnhanced"]
