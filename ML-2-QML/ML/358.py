"""Extended classical fully‑connected layer with configurable depth and regularisation.

The class keeps the original ``run`` API but now supports multiple hidden
layers, dropout, and optional batch‑norm.  Parameters can be supplied as a
flat vector (``thetas``) and are reshaped into the corresponding weight
matrices and biases.  The output is the mean of the final activation, mimicking
the behaviour of the seed example while allowing richer expressivity.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn


class FullyConnectedLayer(nn.Module):
    """Multi‑layer perceptron with optional batch‑norm and dropout.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input vector.
    hidden_sizes : Iterable[int], optional
        Sizes of hidden layers; default ``[64, 32]``.
    dropout : float, optional
        Drop‑out probability; ``0`` disables dropout.
    batchnorm : bool, optional
        Whether to insert a batch‑norm layer after each linear layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_sizes: Iterable[int] | None = None,
        dropout: float = 0.0,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        hidden_sizes = list(hidden_sizes or [64, 32])
        layers: List[nn.Module] = []

        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        # Output head
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _param_shapes(self) -> List[Tuple[int,...]]:
        """Return a list of shapes of all learnable parameters."""
        return [p.shape for p in self.parameters()]

    def set_params_from_vector(self, theta: Iterable[float]) -> None:
        """Flatten a vector of the same length as all parameters and load it."""
        theta = np.asarray(theta, dtype=np.float32).reshape(-1)
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(theta[offset : offset + n]).view_as(p))
            offset += n

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the network with parameters given by ``thetas``.

        The input is a single‑dimensional vector; the network treats it as a
        batch of size one.  The output is a 1‑D ``numpy`` array containing the
        mean value of the final activation, matching the behaviour of the
        original seed implementation.
        """
        self.set_params_from_vector(thetas)
        # Dummy input: a vector of ones matching the expected feature size
        with torch.no_grad():
            input_vec = torch.ones((1, self.net[0].in_features), dtype=torch.float32)
            out = self.forward(input_vec)
            expectation = out.mean(dim=0)
        return expectation.detach().cpu().numpy()

    def num_params(self) -> int:
        """Return the total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())


def FCL(**kwargs) -> FullyConnectedLayer:
    """Convenience factory matching the original API."""
    return FullyConnectedLayer(**kwargs)


__all__ = ["FCL", "FullyConnectedLayer"]
