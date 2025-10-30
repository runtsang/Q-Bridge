"""
A richer classical fully‑connected layer that supports multiple hidden layers,
dropout, batch‑normalisation and convenient weight‑vector loading.

The class implements the same public API as the original seed – a ``run`` method that
accepts an iterable of parameters and returns a single‑element NumPy array with the
mean output of the network.  Internally the module is a standard PyTorch ``nn.Module``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils import vector_to_parameters
from typing import Iterable, Sequence, Optional


class FullyConnectedLayer(nn.Module):
    """
    Fully‑connected neural network with optional hidden layers, dropout and batch‑norm.

    Parameters
    ----------
    n_features:
        Number of input features.
    hidden_sizes:
        Sequence of hidden layer sizes.  An empty sequence yields a single linear layer.
    activation:
        Activation function to use.  Supports 'tanh','relu','sigmoid', 'gelu'.
    dropout:
        Dropout probability applied after each hidden layer (``0.0`` disables dropout).
    batch_norm:
        Whether to insert a batch‑normalisation layer after each hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: Sequence[int] | None = None,
        activation: str = "tanh",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        hidden_sizes = [] if hidden_sizes is None else list(hidden_sizes)

        layers: list[nn.Module] = []
        in_features = n_features
        act = getattr(nn, activation.capitalize())() if activation.lower()!= "gelu" else nn.GELU()

        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = h

        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

    # ---------------------------------------------------------------------------

    def _set_weights_from_vector(self, theta: Iterable[float]) -> None:
        """Load a flat parameter vector into the network."""
        vec = torch.as_tensor(list(theta), dtype=torch.float32)
        vector_to_parameters(vec, self.parameters())

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the network on the given flat parameter vector.

        Parameters
        ----------
        thetas:
            Iterable of floats that matches the number of learnable parameters.

        Returns
        -------
        np.ndarray
            One‑element array containing the mean of the network output.
        """
        self._set_weights_from_vector(thetas)
        with torch.no_grad():
            dummy = torch.zeros((1, self.model[0].in_features))
            out = self.model(dummy)
            mean_out = out.mean().item()
        return np.array([mean_out])

    # ---------------------------------------------------------------------------

    def parameters_vector_length(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


def FCL(
    n_features: int = 1,
    hidden_sizes: Sequence[int] | None = None,
    activation: str = "tanh",
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> FullyConnectedLayer:
    """Convenience factory mirroring the original API."""
    return FullyConnectedLayer(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
    )


__all__ = ["FullyConnectedLayer", "FCL"]
