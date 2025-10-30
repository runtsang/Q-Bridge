"""Enhanced fully‑connected layer for classical experiments.

The new `FCL` class supports multiple hidden layers, dropout, batch‑normalization,
and a flexible activation function.  It inherits from `torch.nn.Module` and
provides a `run` method that accepts a flat list of parameters (weights and
biases) and returns the network output as a NumPy array.  The interface is
compatible with the original seed, allowing drop‑in replacement in existing
workflows.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FCL(nn.Module):
    """Fully‑connected neural network with optional dropout and batch‑norm.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_sizes : Sequence[int], optional
        Sizes of hidden layers.  If omitted, a single hidden layer of size
        ``2 * n_features`` is used.
    activation : str, optional
        Activation function for hidden layers.  Supported values: ``'relu'``,
        ``'tanh'``, ``'sigmoid'``.
    dropout_rate : float, optional
        Dropout probability.  Set to 0.0 to disable dropout.
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: Sequence[int] | None = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [2 * n_features]
        layers = []
        in_dim = n_features
        act_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }[activation]
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(h))
            in_dim = h
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the network with a flat list of parameters.

        Parameters
        ----------
        thetas
            Flattened list of weights and biases in the order produced by
            ``torch.nn.utils.parameters_to_vector``.
        Returns
        -------
        np.ndarray
            Network output as a 1‑D array.
        """
        # Load parameters into the model
        param_vec = torch.tensor(list(thetas), dtype=torch.float32)
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(param_vec, self.parameters())
        # Forward pass on a dummy input of shape (1, n_features)
        dummy = torch.randn(1, self.network[0].in_features, dtype=torch.float32)
        out = self.forward(dummy)
        return out.detach().cpu().numpy().flatten()


__all__ = ["FCL"]
