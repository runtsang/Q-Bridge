"""Hybrid classical QCNN network that mimics the quantum feature extraction
and adds a learnable fully‑connected head.

The architecture is a direct descendant of the original QCNNModel and
EstimatorQNN.  It replaces the final linear head with a small
fully‑connected network that has the same input dimensionality as the
output of the last pooling layer.  The model can be trained with any
PyTorch optimiser and can be used as a drop‑in replacement for the
original QCNNModel in downstream pipelines.

The class name `QCNNHybrid` is kept identical in the quantum module
to facilitate consistent API usage across the two implementations.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable


class QCNNHybrid(nn.Module):
    """
    Classical implementation of the QCNN architecture.

    Layers:
        * feature_map: Linear(8 → 16) + Tanh
        * conv1: Linear(16 → 16) + Tanh
        * pool1: Linear(16 → 12) + Tanh
        * conv2: Linear(12 → 8) + Tanh
        * pool2: Linear(8 → 4) + Tanh
        * conv3: Linear(4 → 4) + Tanh
        * pool3: Linear(4 → 2) + Tanh
        * head: Small FC network identical to EstimatorQNN
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extraction stack
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.pool3 = nn.Sequential(nn.Linear(4, 2), nn.Tanh())

        # Fully‑connected head (same topology as EstimatorQNN)
        self.head = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        return torch.sigmoid(self.head(x))


def QCNNHybrid() -> QCNNHybrid:
    """Factory that returns a fully constructed QCNNHybrid instance."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "QCNNHybrid"]
