"""Hybrid classical fully‑connected layer with advanced architecture.

This module defines a classical neural network that mimics a quantum
fully‑connected layer.  It inherits the simple interface from the
anchor example but adds depth, non‑linearities and a mean‑pooling
output, enabling richer feature extraction.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Iterable

class FCLImpl(nn.Module):
    """
    Classical surrogate for a quantum fully‑connected layer.

    Parameters
    ----------
    n_features : int
        Number of input features (the number of theta parameters).
    hidden_dims : list[int]
        Sizes of hidden layers. Defaults to [8, 4] for a shallow network.
    """

    def __init__(self, n_features: int = 1, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [8, 4]
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=0.1))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass that accepts an iterable of theta parameters and
        returns a single‑dimensional prediction.

        The input is first converted into a tensor, passed through the
        network and the output is mean‑pooled across the batch dimension
        to emulate a quantum expectation value.
        """
        x = torch.tensor(list(thetas), dtype=torch.float32).unsqueeze(-1)
        out = self.net(x)
        return out.mean(dim=0, keepdim=True)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience wrapper that mimics the quantum API: returns a
        NumPy array of the expectation value.
        """
        with torch.no_grad():
            return self.forward(thetas).cpu().numpy()

def FCL() -> type[FCLImpl]:
    """Return the FCLImpl class for compatibility with the anchor API."""
    return FCLImpl

__all__ = ["FCLImpl", "FCL"]
