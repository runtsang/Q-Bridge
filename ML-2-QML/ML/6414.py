"""Enhanced classical CNN with depthwise‑separable convolutions and residual connections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QFCModelEnhanced(nn.Module):
    """A lightweight residual CNN that uses depthwise‑separable convolutions.

    The architecture mirrors the original QuantumNAT CNN but replaces
    the two standard convolutions with a pair of depthwise + pointwise
    layers, and adds a residual connection that bypasses the first block.
    This reduces the number of trainable parameters and encourages feature
    reuse, which is particularly useful when later hybridising with a
    quantum layer.
    """

    def __init__(self) -> None:
        super().__init__()

        # First depthwise‑separable block
        self.ds_block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, groups=1),  # depthwise
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),  # pointwise
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Residual shortcut (1x1 conv to match dimensions)
        self.res_conv = nn.Conv2d(1, 8, kernel_size=1, stride=2, bias=False)

        # Second depthwise‑separable block
        self.ds_block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, groups=8),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual shortcut
        shortcut = self.res_conv(x)

        # First block
        out = self.ds_block1(x)

        # Add residual
        out = out + shortcut
        out = F.relu(out)

        # Second block
        out = self.ds_block2(out)

        # Flatten and fully‑connected head
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return self.norm(out)


__all__ = ["QFCModelEnhanced"]
