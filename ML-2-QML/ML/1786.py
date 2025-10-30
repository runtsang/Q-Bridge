"""FullyConnectedLayer implementation for classical deep learning.

The class accepts an arbitrary number of hidden layers and supports batch
processing. The `run` method maps a flattened list of parameters to the
weights/biases of the network, applies a tanh activation, and returns the
mean prediction as a numpy array. This is a drop‑in replacement for the
simple version in the seed and can be used in hybrid pipelines.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


class FullyConnectedLayer(nn.Module):
    """
    Multi‑layer fully connected network that can be instantiated with any
    number of hidden layers. The constructor stores the architecture
    and builds a list of linear layers and activation functions.
    """

    def __init__(self, input_dim: int = 1, hidden_dims: tuple[int,...] = ()):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dims : tuple of int
            Sizes of hidden layers. The output layer always has size 1.
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        # Build hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Parameters
        ----------
        thetas : Iterable[float]
            Flattened list of all weights and biases in the order they appear
            in the network. The length must match ``self.network`` parameters.

        Returns
        -------
        np.ndarray
            Mean of the network output over a dummy batch.
        """
        thetas = list(thetas)
        # Load parameters into the model
        idx = 0
        for name, param in self.named_parameters():
            numel = param.numel()
            param.copy_(torch.tensor(thetas[idx : idx + numel], dtype=torch.float32).view_as(param))
            idx += numel

        # Run the network on a dummy batch (batch_size=1)
        with torch.no_grad():
            x = torch.zeros(1, self.network[0].in_features)
            out = self.network(x)
            expectation = torch.tanh(out).mean()
            return expectation.numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def FCL(input_dim: int = 1, hidden_dims: tuple[int,...] = ()):
    """Convenience factory matching the original seed API."""
    return FullyConnectedLayer(input_dim, hidden_dims)


__all__ = ["FullyConnectedLayer", "FCL"]
