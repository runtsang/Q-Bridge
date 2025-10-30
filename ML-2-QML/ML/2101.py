"""Enhanced classical QCNN model with residual blocks and dropout."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["QCNNModel", "QCNN"]


class ResidualBlock(nn.Module):
    """A two‑layer residual block with optional batch normalization and dropout."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features) if out_features == in_features else nn.Identity()
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        return F.relu(x + residual)


class QCNNModel(nn.Module):
    """Stacked residual blocks emulating a quantum convolution‑pooling hierarchy."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: tuple[int,...] = (16, 16, 12, 8, 4, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.Tanh()]
        for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(ResidualBlock(in_f, out_f, dropout))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.network(inputs)
        return torch.sigmoid(x)


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()
