"""Enhanced classical QCNN with residual blocks and regularisation.

The model now consists of three residual blocks, each with two linear layers,
batch‑normalisation, and dropout.  The output head uses a sigmoid for binary
classification.  A factory function :func:`QCNN` returns a ready‑to‑train
instance.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two‑layer residual block with batch‑norm and dropout."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        # Add residual connection
        if residual.shape == x.shape:
            x = x + residual
        return self.dropout(x)


class QCNNModel(nn.Module):
    """Depth‑enhanced QCNN with residual blocks and regularisation."""

    def __init__(self) -> None:
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        # Residual blocks
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 32)
        self.res3 = ResidualBlock(32, 32)
        # Final classification head
        self.head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.head(x)


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
