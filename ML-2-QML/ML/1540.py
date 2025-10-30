"""Enhanced classical fully connected layer with dropout and batch support."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FullyConnectedLayer(nn.Module):
    """
    A multi‑feature, dropout‑regularised feed‑forward network that mimics a
    dense layer.  The network accepts a single iterable of parameters
    corresponding to input features and returns the output as a NumPy
    array.

    Parameters
    ----------
    n_features : int, default=1
        Number of input features (length of the theta iterable).
    hidden_units : int, default=16
        Size of the hidden linear layer.
    dropout : float, default=0.0
        Dropout probability applied after the hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_units: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_units, 1),
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward pass of the network.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable containing the input parameters (features).

        Returns
        -------
        np.ndarray
            Output of the network as a 1‑D array.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        output = self.network(values)
        return output.detach().numpy().reshape(-1)


__all__ = ["FullyConnectedLayer"]
