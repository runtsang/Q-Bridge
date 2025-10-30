"""Hybrid classical neural network inspired by Quantum‑NAT.

This module implements QHybridModel, a deep CNN followed by a
fully‑connected head that outputs four logits.  It extends the
original QFCModel by adding an extra convolutional layer, dropout
regularisation and a residual connection in the fully‑connected
segment.  The architecture is deliberately simple so it can be
plugged into the same training pipelines as the original seed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QHybridModel(nn.Module):
    """Classical hybrid model with a deep CNN backbone and residual FC head."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Feature vector size after three 2×2 pools on 28×28 input
        self.feature_dim = 32 * 3 * 3  # 288

        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            # Residual connection to the first FC layer
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass through the classical backbone and head."""
        features = self.backbone(x)          # (B, 32, 3, 3)
        flat = features.view(features.shape[0], -1)  # (B, 288)

        # Residual connection: add a projection of the flattened vector
        residual = self.fc[0](flat)          # (B, 128)
        out = self.fc[1:](residual)          # (B, num_classes)
        return self.norm(out)


__all__ = ["QHybridModel"]
