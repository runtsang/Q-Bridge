# ml_code: FCL__gen245.py

"""
Classical fully‑connected neural network that mirrors the original seed
but adds depth, regularization, and gradient support.

The design keeps the same public API (`run(thetas: Iterable[float]) -> np.ndarray`)
so it can be dropped into any pipeline that expected the seed's `FCL`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


class FCL(nn.Module):
    """
    A multi‑layer feed‑forward network that takes a list of parameters
    (treated as the input vector) and produces a single output.

    Attributes
    ----------
    net : nn.Sequential
        The underlying neural network consisting of linear, ReLU,
        dropout, and linear layers.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.net(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run the network on the provided parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Input values that are reshaped to match the expected
            input dimension of the network.

        Returns
        -------
        np.ndarray
            The network output as a 1‑D NumPy array.
        """
        # Convert iterable to tensor, enforce float32 for consistency
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        # Forward pass
        output = self.forward(values)
        # Detach and convert to NumPy
        return output.detach().cpu().numpy().flatten()


__all__ = ["FCL"]
