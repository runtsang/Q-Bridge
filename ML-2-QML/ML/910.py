"""Enhanced classical fully connected layer with training support.

The module defines a reusable :class:`FCL` neural network that can be
instantiated with an arbitrary number of hidden units.  It exposes both
a standard ``forward`` method for use in PyTorch training loops and a
``run`` convenience wrapper that accepts a 1‑D iterable of input values
and returns the network output as a NumPy array.  Dropout and batch
normalisation are included to improve generalisation and convergence.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FCL(nn.Module):
    """
    Fully‑connected neural network with optional dropout and batch
    normalisation.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input vector.
    hidden_dim : int, default 64
        Size of the hidden layer.
    dropout_prob : float, default 0.1
        Dropout probability applied after the hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_dim: int = 64,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear_in = nn.Linear(n_features, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_features)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch_size, 1)``.
        """
        x = self.linear_in(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        return x

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience wrapper that accepts a 1‑D iterable of input values
        and returns the network output as a NumPy array.

        Parameters
        ----------
        thetas : Iterable[float]
            Input vector to be fed to the network.

        Returns
        -------
        np.ndarray
            Network output with shape ``(1,)``.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            output = self.forward(values)
        return output.detach().numpy()

__all__ = ["FCL"]
