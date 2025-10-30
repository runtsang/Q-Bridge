"""Enhanced classical estimator for regression tasks.

This module defines the EstimatorQNN class, a fully‑connected neural network
with dropout, batch‑normalisation and L2 regularisation.  The network
accepts a 2‑dimensional input and outputs a single scalar.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class EstimatorQNN(nn.Module):
    """A robust feed‑forward regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input features.
    hidden_dims : list[int], default [8, 4]
        Sizes of hidden layers.
    dropout : float, default 0.1
        Dropout probability applied after every hidden layer.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 4]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

    @staticmethod
    def build_default() -> "EstimatorQNN":
        """Convenience constructor that matches the original seed API."""
        return EstimatorQNN()

__all__ = ["EstimatorQNN"]
