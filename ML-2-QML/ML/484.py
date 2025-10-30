"""Enhanced classical QCNN architecture with residual connections and dropout.

This model extends the original 8→16→12→8→4→4→1 flow by adding
residual skips, batch‑normalisation and dropout.  It can be used as a drop‑in
replacement for the original QCNNModel while providing stronger regularisation
and faster convergence.
"""

from __future__ import annotations

import torch
from torch import nn


class QCNNGen121(nn.Module):
    """Classical QCNN with residuals and regularisation."""

    def __init__(self) -> None:
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        # Residual blocks
        self.block1 = ResidualBlock(16, 16, dropout=0.2)
        self.pool1 = nn.Linear(16, 12)
        self.block2 = ResidualBlock(12, 8, dropout=0.2)
        self.pool2 = nn.Linear(8, 4)
        self.block3 = ResidualBlock(4, 4, dropout=0.2)
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.block1(x)
        x = torch.relu(self.pool1(x))
        x = self.block2(x)
        x = torch.relu(self.pool2(x))
        x = self.block3(x)
        return torch.sigmoid(self.head(x))


class ResidualBlock(nn.Module):
    """Small residual block used throughout the QCNN."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def build_QCNNGen121() -> QCNNGen121:
    """Factory that returns a ready‑to‑train :class:`QCNNGen121`."""
    return QCNNGen121()


__all__ = ["QCNNGen121", "build_QCNNGen121"]
