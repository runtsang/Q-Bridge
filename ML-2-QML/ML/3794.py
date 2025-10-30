"""
QCNNHybridModel: Classical convolution‑inspired network with an embedded
regression head.

This implementation combines the stacked linear layers from the original
QCNNModel and the small feed‑forward regressor from EstimatorQNN, producing a
rich, fully‑classical architecture suitable for regression or binary
classification.
"""

from __future__ import annotations

import torch
from torch import nn


class QCNNHybridModel(nn.Module):
    """Full classical QCNN‑style network with a regression head."""

    def __init__(self) -> None:
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        # Convolution / pooling sequence
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Small regression head (mirrors EstimatorQNN)
        self.regressor = nn.Sequential(
            nn.Linear(4, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 1)
        )
        self.output = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.regressor(x)
        return self.output(x)


def QCNNHybrid() -> QCNNHybridModel:
    """Factory returning the fully‑connected QCNN‑style model with embedded head."""
    return QCNNHybridModel()


__all__ = ["QCNNHybrid", "QCNNHybridModel"]
