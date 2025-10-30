"""Enhanced QCNN model with residual connections and dropout.

This module defines a PyTorch implementation of a QCNN-inspired
network that incorporates batch normalisation, dropout, and
skip connections.  The architecture mirrors the original
structure but adds regularisation and a more flexible
feature extractor.

The public factory function `QCNN()` returns a ready‑to‑train
instance of :class:`QCNNModel`.
"""

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["QCNN", "QCNNModel"]

class QCNNModel(nn.Module):
    """QCNN-inspired fully‑connected network with residual blocks."""
    def __init__(self, in_features: int = 8, hidden_sizes=(16, 12, 8, 4), dropout: float = 0.2):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.res_blocks = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.res_blocks.append(block)
        self.head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            # Add residual connection if dimensions match
            if residual.shape[-1] == x.shape[-1]:
                x = x + residual
        x = torch.sigmoid(self.head(x))
        return x

def QCNN() -> QCNNModel:
    """Factory returning a default QCNNModel instance."""
    return QCNNModel()
