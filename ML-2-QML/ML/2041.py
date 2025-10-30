"""Extended classical fully connected layer mimicking a quantum layer.

The class supports an arbitrary number of hidden layers, batch inference,
and both mean and sum aggregations.  It retains the simple ``run`` API
from the original seed while adding useful utilities for experiment
design.
"""

from __future__ import annotations

from typing import Iterable, Sequence, List, Optional

import numpy as np
import torch
from torch import nn


class FullyConnectedLayer(nn.Module):
    """
    A simple feed‑forward network that emulates a quantum fully‑connected
    layer.  The network is fully differentiable and can be trained
    with any PyTorch optimiser.

    Parameters
    ----------
    n_features : int
        Number of input features (length of the theta sequence).
    n_hidden : int, optional
        Number of hidden units.  If zero, the network reduces to a
        single linear layer.
    activation : str, optional
        Activation to use after each hidden layer.  Supported values
        are ``'tanh'`` and ``'relu'``.  Defaults to ``'tanh'``.
    aggregation : str, optional
        Aggregation to apply to the final output.  ``'mean'`` (default)
        or ``'sum'``.  This mirrors the behaviour of the original
        quantum example which returned a single scalar.
    """

    def __init__(
        self,
        n_features: int = 1,
        n_hidden: int = 0,
        activation: str = "tanh",
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.aggregation = aggregation.lower()
        if self.aggregation not in {"mean", "sum"}:
            raise ValueError("aggregation must be'mean' or'sum'")

        layers: List[nn.Module] = []
        input_dim = n_features
        if n_hidden > 0:
            layers.append(nn.Linear(input_dim, n_hidden))
            if activation.lower() == "tanh":
                layers.append(nn.Tanh())
            elif activation.lower() == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError("Unsupported activation")
            input_dim = n_hidden

        layers.append(nn.Linear(input_dim, 1))
        # No activation on the output layer
        self.net = nn.Sequential(*layers)

    def _forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Internal forward pass that returns a raw tensor of shape (1,).
        """
        x = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return self.net(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run a single forward pass and return a NumPy array of shape (1,).
        """
        with torch.no_grad():
            out = self._forward(thetas)
            if self.aggregation == "mean":
                out = out.mean()
            elif self.aggregation == "sum":
                out = out.sum()
        return out.detach().cpu().numpy()

    def run_batch(
        self, batch_thetas: Sequence[Iterable[float]]
    ) -> np.ndarray:
        """
        Run a batch of forward passes.

        Parameters
        ----------
        batch_thetas : Sequence[Iterable[float]]
            Iterable of theta sequences; all sequences must have length
            ``n_features``.

        Returns
        -------
        np.ndarray
            Shape ``(len(batch_thetas),)``.
        """
        outputs = []
        for theta_seq in batch_thetas:
            outputs.append(self.run(theta_seq))
        return np.vstack(outputs)


def FCL(
    n_features: int = 1,
    n_hidden: int = 0,
    activation: str = "tanh",
    aggregation: str = "mean",
) -> FullyConnectedLayer:
    """
    Factory function mirroring the original API.  It returns an instance
    of ``FullyConnectedLayer`` configured with the supplied parameters.

    The function name and signature are kept identical to the seed,
    ensuring backward compatibility for scripts that import ``FCL``.
    """
    return FullyConnectedLayer(
        n_features=n_features,
        n_hidden=n_hidden,
        activation=activation,
        aggregation=aggregation,
    )


__all__ = ["FCL", "FullyConnectedLayer"]
