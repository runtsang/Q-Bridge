"""
Enhanced fully connected layer implementation.

This module defines a classical neural network that expands upon the seed
by allowing arbitrary depth, dropout, and batch‑normalization.  The network
exposes a `run` method that takes a flat list of parameters (weights and
biases) and returns the network’s output as a NumPy array.  This design
mirrors the interface of the quantum counterpart while providing a richer
classical baseline for comparisons.

Usage
-----
>>> from FCL__gen072 import FCL
>>> model = FCL(n_features=3, hidden_sizes=[8, 4], dropout=0.1)
>>> params = model.get_flat_params()
>>> out = model.run(params)
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FullyConnectedLayer(nn.Module):
    """
    Multi‑layer perceptron with optional dropout and batch‑norm.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: List[int] | None = None,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = []

        layers: List[nn.Module] = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the network given a flat list of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flattened weights and biases of the network.

        Returns
        -------
        np.ndarray
            Output of the network as a 1‑D array.
        """
        # Load parameters into the model
        flat_params = torch.tensor(list(thetas), dtype=torch.float32)
        # Ensure the number of parameters matches
        expected = sum(p.numel() for p in self.parameters())
        if flat_params.numel()!= expected:
            raise ValueError(
                f"Expected {expected} parameters, got {flat_params.numel()}"
            )
        # Unflatten and assign
        pointer = 0
        for param in self.parameters():
            num = param.numel()
            param.data.copy_(flat_params[pointer : pointer + num].view_as(param))
            pointer += num

        # Forward pass
        with torch.no_grad():
            output = self.forward(torch.zeros(1, self.network[0].in_features))
        return output.squeeze().detach().numpy()

    def get_flat_params(self) -> np.ndarray:
        """
        Return the current parameters as a flat NumPy array.

        Useful for initializing or comparing with the quantum model.
        """
        return np.concatenate([p.detach().cpu().numpy().flatten() for p in self.parameters()])


__all__ = ["FullyConnectedLayer"]
