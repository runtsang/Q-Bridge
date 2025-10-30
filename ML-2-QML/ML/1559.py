"""Enhanced classical binary classifier with residual blocks and batch‑norm.

This module builds upon the original QCNet architecture, adding
residual connections, batch‑normalisation, and a more expressive
fully‑connected head.  The hybrid head remains a simple sigmoid
layer, but can be swapped for any differentiable activation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two‑layer residual block with batch‑norm and ReLU."""
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


class QCNet(nn.Module):
    """Classical convolutional backbone with a hybrid sigmoid head."""
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.res1 = ResidualBlock(32, 64, stride=2)
        self.res2 = ResidualBlock(64, 128, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

        # Simple sigmoid head used as a stand‑in for the quantum part
        self.head = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.flatten(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        logits = self.fc2(x)
        probs = self.head(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QCNet", "ResidualBlock"]
