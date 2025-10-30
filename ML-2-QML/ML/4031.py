"""Hybrid QCNN model combining convolutional layers with a lightweight regression head.

The architecture mirrors the classical QCNN model but augments it with a small
regression network (inspired by EstimatorQNN) to provide a finer output
resolution.  Dropout and batch‑normalisation layers are added to improve
regularisation and training stability.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class HybridQCNN(nn.Module):
    """Convolution‑inspired feature extractor followed by a regression head."""
    def __init__(self) -> None:
        super().__init__()
        # Feature extraction block (mimics QCNNModel)
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1)
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1)
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.Tanh(),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.Tanh(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.1)
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.BatchNorm1d(4),
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.BatchNorm1d(4),
            nn.Dropout(0.1)
        )
        # Regression head (inspired by EstimatorQNN)
        self.regressor = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.1),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.BatchNorm1d(4),
            nn.Dropout(0.1),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return self.regressor(x)

def HybridQCNN() -> HybridQCNN:
    """Factory returning a configured :class:`HybridQCNN` instance."""
    return HybridQCNN()

__all__ = ["HybridQCNN", "HybridQCNN"]
