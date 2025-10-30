"""Enhanced classical QCNN with residual connections and hybrid loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import relu, sigmoid
from typing import Optional

class ResidualBlock(nn.Module):
    """A shallow residual block that adds a skip connection across two linear layers."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        return x + residual

class QCNNEnhanced(nn.Module):
    """A stack of linear layers with residual connections and a hybrid loss head."""
    def __init__(self, *, depth: int = 3, hidden_size: int = 16, num_classes: int = 1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, hidden_size), nn.Tanh())
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_size, hidden_size) for _ in range(depth)])
        self.pool1 = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        for block in self.res_blocks:
            x = block(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.head(x)
        return sigmoid(x)

    def hybrid_loss(self, outputs: torch.Tensor, targets: torch.Tensor,
                    aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute a hybrid loss combining BCE and optional MSE on auxiliary features."""
        bce = nn.BCELoss()(outputs, targets)
        if aux is not None:
            mse = nn.MSELoss()(aux, targets)
            return bce + 0.5 * mse
        return bce

def QCNNEnhancedFactory() -> QCNNEnhanced:
    """Factory returning a configured QCNNEnhanced model."""
    return QCNNEnhanced(depth=3, hidden_size=16)

__all__ = ["QCNNEnhanced", "QCNNEnhancedFactory"]
