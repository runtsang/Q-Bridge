"""Enhanced classical QCNN model with residual connections and regularisation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """A simple residual block with two linear layers, batch norm and dropout."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        return F.relu(x + residual)


class QCNNModel(nn.Module):
    """Deepened QCNN emulation with residual blocks and dropout."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, depth: int = 3) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        # Build a stack of residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(depth)]
        )

        # Pooling stages
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Tanh())

        # Final classification head
        self.head = nn.Linear(hidden_dim // 4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        for block in self.residual_blocks:
            x = block(x)
        x = self.pool1(x)
        x = self.pool2(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
