"""ResNet‑style classical binary classifier with a residual block and trainable head.

This module extends the original hybrid model by adding a lightweight residual
block and a fully‑connected head that outputs class probabilities.  The
architecture remains compatible with the original `QCNet` interface so that
the new `HybridClassicalHead` can be swapped in without changing the rest of
the pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A simple residual block with two 3×3 convolutions."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut path
        if stride!= 1 or in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)
        return out


class HybridClassicalHead(nn.Module):
    """Trainable head that maps flattened features to binary probabilities."""

    def __init__(self, in_features: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(120, 84)
        self.bn2 = nn.BatchNorm1d(84)
        self.out = nn.Linear(84, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.out(x)
        return F.softmax(x, dim=-1)


class QCNet(nn.Module):
    """Convolutional backbone followed by a hybrid classical head."""

    def __init__(self) -> None:
        super().__init__()
        # Backbone: two conv layers + one residual block
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.resblock = ResidualBlock(6, 12, stride=1)
        # The flattened feature size is 12 * 15 * 15 = 2700 for 32×32 inputs
        self.head = HybridClassicalHead(in_features=2700)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.resblock(x)
        x = torch.flatten(x, 1)
        probs = self.head(x)
        return probs


__all__ = ["ResidualBlock", "HybridClassicalHead", "QCNet"]
