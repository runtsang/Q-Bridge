"""
Enhanced classical fully‑connected layer with depth, dropout, and a simple
`run` interface that accepts a list of input values and returns a scalar
expectation.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from torch import nn


def FCL() -> nn.Module:
    """Return a fully‑connected neural network with optional depth and dropout."""

    class FullyConnectedLayer(nn.Module):
        def __init__(
            self,
            n_features: int = 1,
            hidden_sizes: List[int] | None = None,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if hidden_sizes is None:
                hidden_sizes = [10, 10]
            layers: List[nn.Module] = []
            in_dim = n_features
            for h in hidden_sizes:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.network = nn.Sequential(*layers)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """
            Accept a list of input values, forward them through the network,
            and return the mean of the tanh of the output as a numpy array.
            """
            inputs = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            with torch.no_grad():
                out = self.network(inputs)
                expectation = torch.tanh(out).mean()
                return np.array([expectation.item()])

    return FullyConnectedLayer()


__all__ = ["FCL"]
