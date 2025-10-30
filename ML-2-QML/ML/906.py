"""Enhanced QCNN-inspired classical network with residual connections and adaptive pooling."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two‑layer residual block with optional dimension matching."""
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.skip = nn.Sequential()
        if in_dim!= out_dim:
            self.skip = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)


class QCNNModel(nn.Module):
    """Hybrid QCNN‑inspired network with residual blocks and dropout."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(32, 32) for _ in range(3)]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(32, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        for block in self.residual_blocks:
            x = block(x)
        # Global pooling over the feature dimension
        x = self.global_pool(x.unsqueeze(0)).squeeze(0)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
