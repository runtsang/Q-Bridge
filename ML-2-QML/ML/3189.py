"""Hybrid classical model that emulates quantum convolution + fully‑connected layers."""
from __future__ import annotations

import torch
from torch import nn
from typing import Iterable


class QCNNHybridModel(nn.Module):
    """Stack of fully‑connected layers that mirrors the QCNN convolution + pooling stages."""
    def __init__(self, input_dim: int = 8, fc_features: int = 1) -> None:
        super().__init__()
        # Feature map (input embedding)
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        # Convolution & pooling stages
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.pool3 = nn.Sequential(nn.Linear(4, 2), nn.Tanh())
        # Fully‑connected quantum‑style layer
        self.fc = nn.Linear(2, fc_features)
        # Output head
        self.head = nn.Linear(fc_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = torch.tanh(self.fc(x))
        return torch.sigmoid(self.head(x))

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """Mimic a quantum fully‑connected layer by applying a linear transform to thetas."""
        theta_tensor = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.fc(theta_tensor)).mean()


def QCNN() -> QCNNHybridModel:
    """Factory returning a configured :class:`QCNNHybridModel`."""
    return QCNNHybridModel()


__all__ = ["QCNN", "QCNNHybridModel"]
