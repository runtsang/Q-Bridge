"""Hybrid QCNN-inspired network with residual connections and dropout."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class QCNNHybridModel(nn.Module):
    """A deeper, residual QCNN-inspired architecture with dropout and batchnorm.

    The original seed used a simple stack of linear layers.  Here we
    introduce:
      * Residual blocks that preserve feature dimensionality.
      * Dropout for regularisation and better generalisation.
      * BatchNorm to stabilise training on larger datasets.
    """

    def __init__(self, input_dim: int = 8, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 32, 16, 8]
        self.input_dim = input_dim

        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            block = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dims[i + 1], hidden_dims[i]),
                nn.BatchNorm1d(hidden_dims[i]),
            )
            self.blocks.append(block)

        # Final head
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[0], 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.feature_map(x)
        for block in self.blocks:
            residual = out
            out = block(out)
            out = residual + out  # residual connection
        out = self.head(out)
        return out

def QCNNHybrid() -> QCNNHybridModel:
    """Factory returning the configured :class:`QCNNHybridModel`."""
    return QCNNHybridModel()

__all__ = ["QCNNHybrid", "QCNNHybridModel"]
