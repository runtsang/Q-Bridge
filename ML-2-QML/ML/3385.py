"""Hybrid classical regressor inspired by QCNN and EstimatorQNN.

The network first encodes 2‑dimensional inputs through a small
feed‑forward feature map, then applies three convolution‑pool blocks
mirroring the QCNN architecture, and finally outputs a scalar
regression value.  The architecture is deliberately lightweight
so it can be used as the classical partner of the quantum
estimator.
"""

from __future__ import annotations

import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """Hybrid classical neural network for regression.

    The network is a compact variant of the QCNN model: a feature map
    followed by three convolution‑pool blocks and a sigmoid‑activated
    head.  It can be used independently or as the classical feature
    extractor feeding a quantum circuit.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature map: 2 → 8
        self.feature_map = nn.Sequential(
            nn.Linear(2, 8, bias=False),
            nn.Tanh()
        )
        # Convolution‑pool blocks
        self.conv1 = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(8, 6), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(6, 6), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(6, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Output head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

__all__ = ["EstimatorQNN"]
