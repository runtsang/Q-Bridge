"""Enhanced classical QCNN model with residual connections and regularization."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Two‑layer residual block with batch‑norm and dropout."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(0.2)
        self.residual = (
            nn.Linear(in_features, out_features)
            if in_features!= out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out += residual
        return F.relu(out)

class QCNNModel(nn.Module):
    """Stack of residual blocks that mimics a quantum convolutional network."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(), nn.BatchNorm1d(32)
        )
        self.conv1 = ResidualBlock(32, 32)
        self.pool1 = nn.Sequential(nn.Linear(32, 24), nn.ReLU())
        self.conv2 = ResidualBlock(24, 16)
        self.pool2 = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        self.conv3 = ResidualBlock(8, 8)
        self.head = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return self.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    """Factory returning a ready‑to‑train :class:`QCNNModel`."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
