"""Enhanced classical QCNN with residual connections and dropout."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class QCNNEnhanced(nn.Module):
    """Classical QCNN with residual connections, batch norm, and dropout."""
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        # First block (feature map)
        self.feature_map = nn.Sequential(layers[0:4])
        # Remaining blocks
        self.blocks = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            seq = nn.Sequential(
                nn.Linear(in_dim, hidden_dims[i]),
                nn.BatchNorm1d(hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.blocks.append(seq)
            in_dim = hidden_dims[i]
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual  # residual connection
        logits = self.head(x)
        return torch.sigmoid(logits)

def QCNNEnhanced() -> QCNNEnhanced:
    """Factory returning an initialized QCNNEnhanced."""
    return QCNNEnhanced()

__all__ = ["QCNNEnhanced", "QCNNEnhanced"]
