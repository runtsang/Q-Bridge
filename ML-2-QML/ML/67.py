"""Enhanced classical fully connected layer with dropout and custom initialization."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn

class FullyConnectedLayer(nn.Module):
    """
    A flexible fully‑connected layer that accepts an iterable of input features,
    applies an optional dropout, and uses a user‑selected weight initialization.
    The ``run`` method keeps the original seed API (list of thetas → scalar output).
    """
    def __init__(self, n_features: int = 1, dropout: float = 0.0,
                 init: str = "xavier") -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        if init == "xavier":
            nn.init.xavier_uniform_(self.linear.weight)
        elif init == "kaiming":
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="tanh")
        else:
            raise ValueError(f"Unsupported init {init!r}")
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout:
            x = self.dropout(x)
        return torch.tanh(self.linear(x))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimics the original seed API: ``run`` accepts an iterable of thetas
        (the input features), performs a tanh‑activated linear transformation,
        and returns the mean of the activations as a NumPy array.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.forward(values)
        return out.mean(dim=0).detach().numpy()

def FCL() -> FullyConnectedLayer:
    """
    Compatibility wrapper that returns a default instance of ``FullyConnectedLayer``.
    """
    return FullyConnectedLayer()

__all__ = ["FullyConnectedLayer", "FCL"]
