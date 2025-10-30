"""Quantum-inspired convolutional neural network with modern PyTorch components."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class QCNNModel(nn.Module):
    """A deeper, regularized QCNN-like architecture.

    The model mirrors the original layerwise structure but adds
    BatchNorm, ReLU, and dropout to improve generalisation.
    """

    def __init__(self, input_dim: int = 8, hidden_dims: list[int] | None = None, dropout: float = 0.2) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))

def QCNN() -> QCNNModel:
    """Return a default‑configured QCNNModel."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
