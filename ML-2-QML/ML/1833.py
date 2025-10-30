"""Enhanced classical QCNN model with modern layers and configurable dropout."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating quantum convolution steps.

    The network now includes BatchNorm, ReLU activations, and a Dropout
    layer for regularisation.  A factory accepts a `dropout` hyper‑parameter
    to control regularisation strength.
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        return torch.sigmoid(self.head(x))


def QCNN(dropout: float = 0.2) -> QCNNModel:
    """Factory returning a fully configured :class:`QCNNModel`."""
    return QCNNModel(dropout=dropout)


__all__ = ["QCNN", "QCNNModel"]
