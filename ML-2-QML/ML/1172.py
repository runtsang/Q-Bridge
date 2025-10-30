"""Enhanced fully connected layer with optional hidden layers and activation.

This module defines a flexible neural network that can be used as a drop‑in
replacement for the original `FCL` example.  It supports arbitrary input
dimensionality, multiple hidden layers, dropout, batch‑normalisation and
different activation functions.  The `run` method accepts a flat list of
weights that are reshaped into the module’s parameters, allowing the layer
to be used in optimisation loops that supply raw theta vectors.

The class is compatible with the original API: `FCL()` returns an instance
with a `run` method.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FullyConnectedLayer(nn.Module):
    """A flexible fully‑connected neural network."""

    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: Iterable[int] | None = None,
        activation: str = "tanh",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        n_features:
            Size of the input vector.
        hidden_sizes:
            Sequence of hidden layer sizes.  If ``None`` a single linear layer
            is used (the original behaviour).
        activation:
            Activation function for hidden layers.  Supported values are
            ``"tanh"``, ``"relu"``, ``"sigmoid"`` and ``"gelu"``.
        dropout:
            Dropout probability applied after each hidden layer.
        batch_norm:
            Whether to apply a BatchNorm1d after each hidden layer.
        """
        super().__init__()

        layers = []
        in_size = n_features
        act_fn = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "gelu": F.gelu,
        }[activation]

        if hidden_sizes is None:
            hidden_sizes = []

        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_size = h

        # Output layer
        layers.append(nn.Linear(in_size, 1))
        # No activation on the output – the caller can apply any non‑linearity
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Load a flat list of parameters into the network and return the
        mean output over a batch of unit‑length inputs.

        The shape of ``thetas`` must match the total number of learnable
        parameters in the network.  The method is intentionally lightweight
        so it can be called from optimisation loops that only supply raw
        theta vectors.
        """
        flat = torch.tensor(list(thetas), dtype=torch.float32)
        # Assign parameters
        idx = 0
        for param in self.parameters():
            num = param.numel()
            param.data.copy_(flat[idx : idx + num].view_as(param))
            idx += num

        # Create a dummy batch of unit‑length vectors
        batch = torch.ones((100, self.model[0].in_features))
        output = self.forward(batch).mean(dim=0)
        return output.detach().cpu().numpy()


def FCL() -> FullyConnectedLayer:
    """Return a fully‑connected layer with a single hidden layer and tanh."""
    return FullyConnectedLayer(n_features=1, hidden_sizes=[10], activation="tanh")


__all__ = ["FCL"]
