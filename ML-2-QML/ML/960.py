"""HybridEstimator implementation for classical neural networks.

This module defines a flexible feed‑forward regressor that supports variable
input dimensionality, a list of hidden layers, dropout and batch
normalization.  It can be used as a drop‑in replacement for the original
EstimatorQNN while offering richer expressivity.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence

class HybridEstimator(nn.Module):
    """A configurable feed‑forward regression network.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : Sequence[int], optional
        Sizes of hidden layers.  Defaults to (8, 4).
    output_dim : int, optional
        Size of the output layer.  Defaults to 1.
    dropout : float, optional
        Dropout probability applied after each hidden layer.  Defaults to 0.0.
    batch_norm : bool, optional
        Whether to insert batch‑norm after each hidden layer.  Defaults to False.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        output_dim: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or (8, 4)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

__all__ = ["HybridEstimator"]
