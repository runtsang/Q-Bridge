"""Enhanced fully connected layer with multi‑layer MLP and dropout.

The class accepts a flattened list of input parameters, feeds them through a
deep neural network, and returns the mean tanh of the final output.  It
mirrors the original interface while adding depth and regularisation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List


class FCL(nn.Module):
    """
    Multi‑layer fully connected network that processes a parameter vector.

    Parameters
    ----------
    n_features : int, default 1
        Number of input features (size of the parameter vector).
    hidden_sizes : List[int], default [10]
        Sizes of hidden layers.
    output_size : int, default 1
        Size of the output layer.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: List[int] | None = None,
        output_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [10]
        layers: List[nn.Module] = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.network = nn.Sequential(*layers)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward pass using the supplied parameter vector.

        Parameters
        ----------
        thetas : Iterable[float]
            Input parameter vector of length ``n_features``.

        Returns
        -------
        np.ndarray
            Mean tanh of the final output, wrapped in a 1‑D array.
        """
        values = torch.tensor(list(thetas), dtype=torch.float32).view(1, -1)
        out = self.network(values)
        expectation = torch.tanh(out).mean().item()
        return np.array([expectation])


__all__ = ["FCL"]
