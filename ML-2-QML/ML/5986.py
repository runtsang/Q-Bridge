"""
EstimatorQNN: An extensible classical regression model.

This module builds on the original toy network by exposing a flexible
architecture that can be tuned via the *hidden_sizes* argument.
Dropout and batch‑norm are optional, allowing users to experiment with
regularisation without changing the public API.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence, Iterable

__all__ = ["EstimatorQNN"]


def EstimatorQNN(
    hidden_sizes: Sequence[int] | None = None,
    dropout: float | None = None,
    batch_norm: bool = False,
) -> nn.Module:
    """
    Return a fully‑connected regression network with optional regularisation.

    Parameters
    ----------
    hidden_sizes : Sequence[int] | None
        Sizes of hidden layers.  If ``None`` (default) a minimal 2 → 8 → 4 → 1
        network is built, matching the original seed.
    dropout : float | None
        Dropout probability applied after each hidden layer.  If ``None`` no
        dropout is added.
    batch_norm : bool
        Whether to insert a BatchNorm1d layer after each hidden layer.

    Returns
    -------
    nn.Module
        Instantiated network ready for training.
    """
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if hidden_sizes is None:
                hidden_sizes = [8, 4]

            layers: Iterable[nn.Module] = []
            in_features = 2
            for size in hidden_sizes:
                layers.append(nn.Linear(in_features, size))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(size))
                layers.append(nn.Tanh())
                if dropout is not None:
                    layers.append(nn.Dropout(dropout))
                in_features = size

            layers.append(nn.Linear(in_features, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs)

    return EstimatorNN()
