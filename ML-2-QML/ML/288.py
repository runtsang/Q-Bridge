"""Enhanced QCNN with residual connections and regularization."""

from __future__ import annotations

import torch
from torch import nn


class QCNNGen321Model(nn.Module):
    """Quantum‑inspired CNN with residuals, batch‑norm and dropout."""

    def __init__(self) -> None:
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        # Convolutional layers with residuals
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(16, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pool1 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual after feature map
        residual = x
        x = self.feature_map(x)
        x = x + residual

        # Convolutional layers with residuals
        residual = x
        x = self.conv1(x)
        x = x + residual

        residual = x
        x = self.conv2(x)
        x = x + residual

        residual = x
        x = self.conv3(x)
        x = x + residual

        # Pooling and final head
        x = self.pool1(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNGen321Model:
    """Factory returning the configured QCNNGen321Model."""
    return QCNNGen321Model()


__all__ = ["QCNN", "QCNNGen321Model"]
