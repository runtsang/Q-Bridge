"""Enhanced fully connected layer for classical experiments.

The class implements a small feed‑forward network with configurable
hidden depth, dropout, and batch‑norm.  The public ``run`` method
accepts a sequence of input values (``thetas``) and returns the
network output as a NumPy array, mimicking the original API while
offering richer behaviour for downstream experiments.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FCL(nn.Module):
    """Feed‑forward network with optional hidden layers.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_units : Sequence[int] | None
        Sizes of hidden layers.  ``None`` creates a single linear layer.
    dropout : float
        Drop‑out probability applied after each hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_units: Sequence[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_units is None:
            hidden_units = []

        layers = []
        in_features = n_features
        for out_features in hidden_units:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features
        # Final output layer
        layers.append(nn.Linear(in_features, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.net(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the network on a list of input values.

        Parameters
        ----------
        thetas : Iterable[float]
            Input values; they are reshaped to ``(-1, 1)`` before being fed
            into the network.

        Returns
        -------
        np.ndarray
            The network output as a 1‑D NumPy array.
        """
        with torch.no_grad():
            inputs = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            out = self(inputs)
            # Use tanh non‑linearity to keep outputs in (-1, 1)
            out = torch.tanh(out)
            return out.squeeze().numpy()


__all__ = ["FCL"]
