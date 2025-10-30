"""Enhanced classical regression model with configurable depth and optional dropout."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, List, Optional

class EstimatorNN(nn.Module):
    """
    Feed‑forward regression network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input.
    hidden_sizes : Iterable[int]
        Sizes of hidden layers. Defaults to (8, 4).
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    activation : nn.Module, optional
        Activation function. Defaults to nn.Tanh().
    output_dim : int, default 1
        Size of the output.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Iterable[int] = (8, 4),
        dropout: float = 0.0,
        activation: nn.Module = nn.Tanh(),
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = size
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

def EstimatorQNN(
    input_dim: int = 2,
    hidden_sizes: Iterable[int] = (8, 4),
    dropout: float = 0.0,
    activation: nn.Module = nn.Tanh(),
    output_dim: int = 1,
) -> EstimatorNN:
    """Factory that returns a configured EstimatorNN instance."""
    return EstimatorNN(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        activation=activation,
        output_dim=output_dim,
    )

__all__ = ["EstimatorNN", "EstimatorQNN"]
