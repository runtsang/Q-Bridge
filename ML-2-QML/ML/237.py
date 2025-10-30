"""
Enhanced classical feed‑forward regressor with configurable depth, dropout and batch‑normalisation.
"""

from __future__ import annotations

import torch
from torch import nn


def EstimatorQNN(
    input_dim: int = 2,
    hidden_dims: list[int] | tuple[int,...] = (64, 32),
    dropout: float = 0.1,
    bias: bool = True,
) -> nn.Module:
    """
    Construct a fully‑connected regression network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dims : sequence of int
        Sizes of the hidden layers.  The list length determines the depth.
    dropout : float
        Drop‑out probability applied after each hidden layer.
    bias : bool
        Whether the linear layers include a bias term.

    Returns
    -------
    nn.Module
        Instantiated network ready for training.
    """
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            in_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, h, bias=bias))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, 1, bias=bias))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(x)

    return EstimatorNN()


__all__ = ["EstimatorQNN"]
