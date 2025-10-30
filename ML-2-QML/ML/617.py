"""Extended QCNN model with residual connections and dropout."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with two linear layers, batch norm and ReLU."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        return F.relu(x + residual)


class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating the quantum convolution steps
    with residual connections, batch‑norm, dropout and a final sigmoid head."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: tuple[int, int, int] = (16, 12, 8),
        dropout: float = 0.3,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_block1 = ResidualBlock(hidden_dims[0], hidden_dims[0], dropout)
        self.res_block2 = ResidualBlock(hidden_dims[0], hidden_dims[1], dropout)
        self.res_block3 = ResidualBlock(hidden_dims[1], hidden_dims[2], dropout)
        self.head = nn.Linear(hidden_dims[2], out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.head(x)
        return torch.sigmoid(x)


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
