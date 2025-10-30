"""Classical hybrid binary classifier with residual backbone.

This module implements a fully classical neural network that mimics the
interface of the original QCNet but replaces the quantum head with a
trainable MLP.  The architecture is deliberately deeper to demonstrate
the extension scaling paradigm while keeping the public API identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple 2‑D residual block with batch‑norm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class HybridBinaryClassifier(nn.Module):
    """Hybrid classifier with a residual CNN backbone and an MLP head."""

    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift

        # Residual backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.flatten = nn.Flatten()
        # MLP head
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for a batch of images."""
        x = self.features(x)
        x = self.flatten(x)
        logits = self.head(x)
        probs = torch.sigmoid(logits + self.shift)
        return probs


__all__ = ["HybridBinaryClassifier"]
