"""Combined classical implementation of a fully connected layer with configurable depth.

The class mirrors the classical side of the quantum classifier while exposing
metadata identical to the quantum construction: encoding indices, weight sizes
and observable indices.  The `run` method accepts a list of parameters
(theta values) and returns the mean of a tanh‑activated linear layer,
matching the behaviour of the original FCL seed.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from torch import nn


class FCL(nn.Module):
    """
    Classical fully‑connected network with configurable depth.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int, default 1
        Number of hidden layers.  Each hidden layer is a linear
        transformation followed by a ReLU.  The final layer maps to a
        single output neuron.
    """

    def __init__(self, num_features: int = 1, depth: int = 1) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        # Hidden layers
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

        # Metadata to match the quantum implementation
        self.encoding: List[int] = list(range(num_features))
        self.weight_sizes: List[int] = [
            layer.weight.numel() + layer.bias.numel()
            for layer in self.network.modules()
            if isinstance(layer, nn.Linear)
        ]
        self.observables: List[int] = [0]  # single output neuron

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the network expectation for a list of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flattened list of linear parameters.  The first `num_features`
            values are used as input features and the remaining values are
            ignored (they are not needed for the toy example).

        Returns
        -------
        np.ndarray
            Array of shape (1,) containing the mean tanh activation.
        """
        # Convert to tensor and reshape to column vector
        x = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        # Forward pass
        out = self.network(x)
        # Mean over batch dimension
        expectation = torch.tanh(out).mean(dim=0)
        return expectation.detach().numpy()

    def parameters_flatten(self) -> List[float]:
        """Return all trainable parameters as a flat list."""
        return [p.detach().cpu().numpy().flatten() for p in self.parameters()]


def FCL_factory(num_features: int = 1, depth: int = 1) -> FCL:
    """Convenience factory mirroring the original function interface."""
    return FCL(num_features, depth)


__all__ = ["FCL", "FCL_factory"]
