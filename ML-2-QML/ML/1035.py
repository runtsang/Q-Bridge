"""Enhanced classical QCNN with residuals, batchnorm, and dropout."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """A simple residual block that adds the input to a linear transformation."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        # If dimensions differ, project input
        if x.shape[-1]!= out.shape[-1]:
            x = nn.Linear(x.shape[-1], out.shape[-1]).to(x.device)(x)
        return self.relu(out + x)


class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating quantum convolution steps with residuals."""
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv1 = ResidualBlock(16, 16, dropout)
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv2 = ResidualBlock(12, 8, dropout)
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv3 = ResidualBlock(4, 4, dropout)
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN(dropout: float = 0.1) -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel(dropout=dropout)


__all__ = ["QCNN", "QCNNModel"]
