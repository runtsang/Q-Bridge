"""Hybrid classical QCNN model that emulates quantum convolution steps."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Iterable

__all__ = ["QCNNHybrid", "QCNN"]


class FullyConnectedLayer(nn.Module):
    """
    Classical stand‑in for a quantum fully‑connected layer.

    The layer applies a linear map followed by a tanh non‑linearity and
    returns the mean of the activations, mimicking the behaviour of the
    quantum FCL example.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation


class QCNNHybrid(nn.Module):
    """
    Classical convolution‑inspired network that mirrors the quantum QCNN.

    The architecture follows the original ML seed:
        * Feature map: Linear(8 → 16) + Tanh
        * Convolutional layers: Linear(16 → 16), (12 → 8), (4 → 4)
        * Pooling layers: Linear(16 → 12), (8 → 4)
        * Fully connected head: Linear(4 → 1)
    A dropout layer is added after each convolution to improve generalisation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Dropout(0.1))
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh(), nn.Dropout(0.1))
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh(), nn.Dropout(0.1))
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Dropout(0.1))
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Dropout(0.1))
        self.head = nn.Linear(4, 1)
        self.fcl = FullyConnectedLayer(1)  # quantum‑inspired head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.head(x)
        # Apply the quantum‑inspired fully connected layer
        x = self.fcl(x.squeeze(1))
        return torch.sigmoid(x)


def QCNN() -> QCNNHybrid:
    """
    Factory that returns a fully configured :class:`QCNNHybrid` instance.
    """
    return QCNNHybrid()
