"""Enhanced classical model with residual block and feature fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block with two 3×3 convs and an optional 1×1 shortcut."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (stride!= 1 or in_channels!= out_channels) and nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) or nn.Identity()

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


class QuantumNATEnhanced(nn.Module):
    """Hybrid model: classical CNN with residuals + quantum embedding fusion."""
    def __init__(self, num_classes: int = 4, fusion_dim: int = 16):
        super().__init__()
        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(8, 16, stride=1),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # Fusion linear layer
        self.fusion = nn.Linear(16 * 7 * 7 + fusion_dim, 64)
        self.classifier = nn.Linear(64, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor, quantum_embedding: torch.Tensor) -> torch.Tensor:
        # Classical path
        class_feat = self.encoder(x)
        # Concatenate quantum embedding
        combined = torch.cat([class_feat, quantum_embedding], dim=1)
        out = self.fusion(combined)
        out = F.relu(out)
        out = self.classifier(out)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
