"""Hybrid QCNN model with residual connections and input normalization."""

from __future__ import annotations

import torch
from torch import nn

class QCNNHybrid(nn.Module):
    """
    A hybrid classical QCNN model that extends the original fully connected
    architecture with input normalization and a residual MLP head.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 1) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dim, 12),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.Tanh()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.Tanh()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh()
        )
        # Residual MLP head
        self.residual = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU()
        )
        self.head = nn.Linear(4, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with input normalization and residual connection.
        """
        x = self.input_norm(x)
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        res = self.residual(x)
        x = x + res
        return torch.sigmoid(self.head(x))

def QCNNHybridModel() -> QCNNHybrid:
    """Factory returning a configured :class:`QCNNHybrid`."""
    return QCNNHybrid()

__all__ = ["QCNNHybrid", "QCNNHybridModel"]
