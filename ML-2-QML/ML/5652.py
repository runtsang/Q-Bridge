"""Quantum‑inspired convolutional neural network with residuals and dropout."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A residual block that mimics a quantum convolution step."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.residual = nn.Linear(in_features, out_features) if in_features!= out_features else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + self.residual(x)
        return out

class QCNNModel(nn.Module):
    """A fully‑connected network that emulates a QCNN architecture.

    The network consists of:
    - A feature map (linear + BN + ReLU)
    - Three residual convolutional blocks
    - Two dimensionality‑reduction (pooling) layers implemented with linear layers
    - A final sigmoid classifier
    """
    def __init__(self, input_dim: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv1 = ResidualBlock(16, 16, dropout)
        self.pool1 = nn.Linear(16, 12)
        self.conv2 = ResidualBlock(12, 8, dropout)
        self.pool2 = nn.Linear(8, 4)
        self.conv3 = ResidualBlock(4, 4, dropout)
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = F.max_pool1d(x.unsqueeze(1), kernel_size=2).squeeze(1)  # mimic pooling
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.max_pool1d(x.unsqueeze(1), kernel_size=2).squeeze(1)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def QCNN(input_dim: int = 8, dropout: float = 0.1) -> QCNNModel:
    """Factory that returns a configured :class:`QCNNModel`."""
    return QCNNModel(input_dim, dropout)

__all__ = ["QCNN", "QCNNModel"]
