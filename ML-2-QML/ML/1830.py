"""Classical CNN+MLP model extending Quantum‑NAT.

The network now contains two convolutional blocks, a residual
connection, and a dropout‑regularised classifier, offering higher
representational capacity while remaining fully classical.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class QuantumNATPlus(nn.Module):
    """CNN backbone with residual block, dropout, and MLP head."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Residual block
        self.residual = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
        )
        self.relu = nn.ReLU(inplace=True)
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features -> residual -> classifier.
        :param x: Tensor of shape (batch, 1, 28, 28).
        :return: Normalized logits of shape (batch, num_classes).
        """
        feats = self.features(x)
        res = self.residual(feats)
        feats = self.relu(feats + res)
        logits = self.classifier(feats)
        return self.norm(logits)

__all__ = ["QuantumNATPlus"]
