"""QCNNGen199 – Classical residual convolutional network with dropout and batch‑norm.

This module defines a PyTorch model that mirrors the structure of the original QCNN but
adds residual connections, batch‑normalisation and dropout layers for better
generalisation.  The factory function ``QCNNGen199`` returns a ready‑to‑train
instance.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """A simple residual block with two linear layers, batch‑norm and ReLU."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out = out + residual  # residual connection
        return F.relu(out)


class QCNNGen199(nn.Module):
    """Classical QCNN with residual blocks, dropout and batch‑norm."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU()
        )
        self.res1 = ResidualBlock(16, 16, dropout)
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.BatchNorm1d(12), nn.ReLU())
        self.res2 = ResidualBlock(12, 12, dropout)
        self.pool2 = nn.Sequential(nn.Linear(12, 8), nn.BatchNorm1d(8), nn.ReLU())
        self.res3 = ResidualBlock(8, 8, dropout)
        self.head = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.res3(x)
        return torch.sigmoid(self.head(x))


def QCNNGen199(dropout: float = 0.2) -> QCNNGen199:
    """Factory returning a configured :class:`QCNNGen199`."""
    return QCNNGen199(dropout)


__all__ = ["QCNNGen199", "QCNNGen199"]
