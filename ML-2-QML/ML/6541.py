"""
HybridQuantumCNN – Classical PyTorch module with a residual CNN backbone and a sigmoid head.

The architecture mirrors the original QCNet but augments the feature extractor with
two residual blocks, dropout, and a flexible sigmoid head that can be tuned with a
shift parameter.  The design is fully compatible with the seed model and can be
plugged into existing training pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two‑layer residual block with BatchNorm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class HybridFunction(nn.Module):
    """Differentiable sigmoid head with an optional shift."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)


class HybridQuantumCNN(nn.Module):
    """ResNet‑style CNN followed by a quantum‑aware sigmoid head."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.res1 = ResidualBlock(64, 128, stride=2)
        self.res2 = ResidualBlock(128, 256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)
        self.hybrid = HybridFunction(shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.res1(x)
        x = self.res2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        probs = self.hybrid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["ResidualBlock", "HybridFunction", "HybridQuantumCNN"]
