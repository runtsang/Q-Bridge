"""Enhanced classical fully‑connected block with dropout, batch‑norm and configurable depth.

The module exposes a ``FullyConnectedLayer`` class that can be instantiated with
arbitrary hidden‑layer sizes.  The ``run`` method accepts an iterable of input
vectors and returns the mean activation of the final linear layer – a behaviour
that mirrors the original example while providing a richer, trainable network.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence, List


class FullyConnectedLayer(nn.Module):
    """A small MLP with optional dropout and batch‑normalisation.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input vectors.
    hidden_layers : Sequence[int], optional
        Sizes of the hidden layers.  If omitted a single hidden layer of size
        ``n_features`` is used.
    dropout : float, optional
        Drop‑out probability applied after every activation.  ``0.0`` disables
        it.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_layers: Sequence[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [n_features]
        layers: List[nn.Module] = []

        # Input layer
        prev = n_features
        for size in hidden_layers:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = size

        # Output layer
        layers.append(nn.Linear(prev, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def run(self, thetas: Iterable[Sequence[float]]) -> np.ndarray:
        """
        Evaluate the network on a batch of input vectors.

        Parameters
        ----------
        thetas
            An iterable where each element is an input vector to the network.
            All vectors must be of length ``n_features``.
        Returns
        -------
        np.ndarray
            Mean output of the network over the batch.
        """
        batch = torch.tensor(
            [list(t) for t in thetas], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            output = self(batch).mean(dim=0)
        return output.cpu().numpy()

    @property
    def device(self) -> torch.device:
        """Return the device on which the model parameters reside."""
        return next(self.parameters()).device


def FCL(*args, **kwargs) -> FullyConnectedLayer:
    """Convenience factory mimicking the original seed."""
    return FullyConnectedLayer(*args, **kwargs)


__all__ = ["FullyConnectedLayer", "FCL"]
