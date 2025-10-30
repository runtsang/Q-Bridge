"""Enhanced classical QCNN with residuals, dropout, and batch‑norm."""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class ResidualBlock(nn.Module):
    """A simple residual block with linear, batch‑norm, ReLU, and dropout."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # If dimensions differ, project the shortcut
        self.shortcut = nn.Sequential()
        if in_features!= out_features:
            self.shortcut = nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.relu(out + self.shortcut(x))


class QCNNModel(nn.Module):
    """Stack of residual blocks mimicking quantum convolutional layers."""
    def __init__(self, input_dim: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Residual convolutional blocks
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
    """Factory returning a configured :class:`QCNNModel`."""
    return QCNNModel(dropout=dropout)


__all__ = ["QCNNModel", "QCNN"]
