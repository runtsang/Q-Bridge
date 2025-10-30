"""
Classical implementation of a fully‑connected layer with optional dropout,
batch‑normalisation and multi‑output support.  The class can be instantiated
with a list of parameters that are interpreted as the weights of a single
linear layer.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FCL(nn.Module):
    """
    Fully‑connected layer with configurable regularisation.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_outputs : int, default 1
        Number of output neurons.
    dropout : float, default 0.0
        Dropout probability applied after the linear transform.
    batchnorm : bool, default False
        Whether to include a batch‑normalisation layer.
    """

    def __init__(
        self,
        n_features: int = 1,
        n_outputs: int = 1,
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.batchnorm = nn.BatchNorm1d(n_outputs) if batchnorm else nn.Identity()

        # initialise weights to allow deterministic behaviour
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def update_params(self, thetas: Sequence[float]) -> None:
        """
        Load a flat list of parameters into the linear layer.

        The list must contain ``n_features * n_outputs + n_outputs`` values
        (weights followed by biases).  If the length does not match, an
        exception is raised.
        """
        expected = self.linear.weight.numel() + self.linear.bias.numel()
        if len(thetas)!= expected:
            raise ValueError(
                f"Expected {expected} parameters, got {len(thetas)}."
            )
        weights = torch.tensor(thetas[: self.linear.weight.numel()], dtype=torch.float32)
        biases = torch.tensor(thetas[self.linear.weight.numel() :], dtype=torch.float32)
        self.linear.weight.data = weights.view_as(self.linear.weight)
        self.linear.bias.data = biases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.  The input ``x`` should be a 2‑D tensor
        of shape (batch_size, n_features).
        """
        out = self.linear(x)
        out = self.batchnorm(out)
        out = self.dropout(out)
        return out

    def run(self, thetas: Iterable[float], batch_size: int = 1) -> np.ndarray:
        """
        Execute a forward pass with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat list of weights and biases for the linear layer.
        batch_size : int, default 1
            Number of identical samples to feed through the network.

        Returns
        -------
        np.ndarray
            Mean activation over the batch, shape (n_outputs,).
        """
        self.update_params(thetas)
        # create dummy input of ones to mimic a batch
        x = torch.ones((batch_size, self.linear.in_features), dtype=torch.float32)
        with torch.no_grad():
            out = self.forward(x)
        return out.mean(dim=0).detach().numpy()


__all__ = ["FCL"]
