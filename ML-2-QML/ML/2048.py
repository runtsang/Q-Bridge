"""Enhanced classical fully‑connected layer with optional dropout and bias handling.
The class mirrors the interface of the original quantum layer while adding trainable
weights, bias control, and dropout for richer experimental setups."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


class FCLLayer(nn.Module):
    """
    A PyTorch module that implements a single‑output fully‑connected layer
    with optional dropout and bias. The forward pass applies a tanh
    activation and returns the mean activation as a NumPy array, matching
    the interface of the quantum counterpart.
    """

    def __init__(self, n_features: int = 1, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.linear(values)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out.mean(dim=0)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the mean tanh activation as a NumPy array."""
        return self.forward(thetas).detach().numpy()


def FCL() -> FCLLayer:
    """Convenience factory matching the original API."""
    return FCLLayer()


__all__ = ["FCLLayer", "FCL"]
