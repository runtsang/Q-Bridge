"""
Advanced classical estimator with residual‑style architecture and dropout.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class EstimatorQNNAdvanced(nn.Module):
    """
    A robust regression network that extends the original 2‑layer model.
    Features:
        * Multiple hidden layers with configurable dimensions.
        * Batch normalization and ReLU activations.
        * Dropout for regularisation.
        * Optional residual connections for deeper stacks.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None,
                 dropout: float = 0.2, use_residual: bool = False) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self.use_residual = use_residual
        if self.use_residual:
            # Simple residual skip from input to first hidden layer
            self.residual = nn.Linear(input_dim, hidden_dims[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            x = x + self.residual(x)
        return self.net(x)

def EstimatorQNNAdvanced() -> EstimatorQNNAdvanced:
    """
    Factory returning a ready‑to‑train instance of the advanced estimator.
    """
    return EstimatorQNNAdvanced(use_residual=True)

__all__ = ["EstimatorQNNAdvanced"]
