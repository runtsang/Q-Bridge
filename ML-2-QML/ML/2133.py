"""Hybrid classical CNN for 4‑class classification.

Features:
* Three residual convolutional blocks with skip connections.
* Global average pooling followed by a dropout‑regularised linear head.
* Batch‑norm on the final logits for stable gradients.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQuantumNATModel(nn.Module):
    """Deep residual CNN with attention‑style head."""

    def __init__(self) -> None:
        super().__init__()
        # Residual block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
        )
        # Residual block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
        )
        # Residual block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Residual block 1
        res = self.conv1(x)
        x = res + x
        # Residual block 2
        res = self.conv2(x)
        x = res + x
        # Residual block 3
        res = self.conv3(x)
        x = res + x

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return self.norm(logits)


__all__ = ["HybridQuantumNATModel"]
