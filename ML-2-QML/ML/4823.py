"""Enhanced QCNN model combining classical convolution and fully connected layers with residual pooling.

This module provides a classical neural network architecture that mirrors the quantum QCNN
construction but enriches it with residual connections, adaptive pooling and a
fully connected output head. The design fuses the layer organization from the
original QCNN seed with the depth‑controlled linear stack from
QuantumClassifierModel and the flexible feature‑map idea from FCL.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A simple residual block: Linear → ReLU → Linear + skip."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return F.relu(out + residual)

class QCNNGenModel(nn.Module):
    """Classical network inspired by the quantum QCNN architecture.

    The network consists of an initial feature map followed by a sequence of
    convolution‑like fully‑connected layers, adaptive pooling steps and a
    residual block. The final head maps the reduced representation to a
    single probability.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, depth: int = 3) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        # Build a stack of conv/pool pairs
        self.layers = nn.ModuleList()
        in_dim = hidden_dim
        for i in range(depth):
            conv = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
            pool = nn.Linear(hidden_dim, max(1, hidden_dim // 2))
            self.layers.append(conv)
            self.layers.append(pool)
            in_dim = pool.out_features

        # Residual block after the conv‑pool stack
        self.residual = ResidualBlock(in_dim)

        # Final classifier head
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        for layer in self.layers:
            x = layer(x)
        x = self.residual(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNGenModel:
    """Factory returning a freshly configured QCNNGenModel."""
    return QCNNGenModel()

__all__ = ["QCNN", "QCNNGenModel"]
