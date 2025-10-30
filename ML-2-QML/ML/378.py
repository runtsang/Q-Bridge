"""Enhanced classical fully‑connected layer with configurable depth and dropout."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn
import numpy as np


def FCL() -> nn.Module:
    """Return a fully‑connected neural network that can be seeded by a list of parameters.

    The network is a small MLP with optional hidden layers and dropout, designed
    to mirror the structure of a quantum circuit for fair comparison in hybrid
    experiments.
    """
    return FullyConnectedLayer()


class FullyConnectedLayer(nn.Module):
    """A multi‑layer perceptron with ReLU activations and dropout.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input feature vector.
    hidden_sizes : List[int], optional
        Sizes of hidden layers. Default is two hidden layers of 32 and 16 units.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: List[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [32, 16]
        layers: List[nn.Module] = []
        in_features = n_features

        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = h

        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

    def set_parameters(self, thetas: Iterable[float]) -> None:
        """Overwrite the network parameters with a flat list of values.

        The list length must match the total number of learnable parameters.
        """
        flat_params = torch.cat(
            [p.data.view(-1) for p in self.model.parameters()]
        )
        thetas = torch.tensor(list(thetas), dtype=flat_params.dtype)
        if thetas.shape!= flat_params.shape:
            raise ValueError(
                f"Expected {flat_params.shape[0]} parameters, got {thetas.shape[0]}"
            )
        idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(thetas[idx : idx + numel].view_as(p))
            idx += numel

    def run(self, thetas: Iterable[float] | None = None) -> np.ndarray:
        """Forward pass using the current parameters.

        If *thetas* is supplied, the network weights are first overwritten
        with these values.  The input is a single sample constructed from
        *thetas* when provided, otherwise a zero vector is used.
        """
        if thetas is not None:
            self.set_parameters(thetas)

        x = (
            torch.tensor([list(thetas)], dtype=torch.float32)
            if thetas is not None
            else torch.zeros((1, self.model[0].in_features))
        )
        out = self.model(x)
        return out.detach().numpy()


__all__ = ["FCL"]
