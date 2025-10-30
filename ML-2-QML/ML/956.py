"""Enhanced feed‑forward regressor with residual‑like connections and regularisation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class EstimatorNN(nn.Module):
    """A deeper, regularised regressor that supports feature scaling and dropout."""
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 32, 16]
        layers = []
        prev_dim = input_dim
        for idx, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.2))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def EstimatorQNN() -> EstimatorNN:
    """Return an instance of the enhanced regressor."""
    return EstimatorNN()

__all__ = ["EstimatorQNN"]
