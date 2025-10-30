"""Extended EstimatorQNN with residual connections, batch‑normalization, and dropout for improved regression."""

from __future__ import annotations

import torch
from torch import nn


class EstimatorNN(nn.Module):
    """
    A small but more expressive feed‑forward network.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input features.
    hidden_dims : list[int], default [8, 4]
        Sizes of hidden layers.
    dropout : float, default 0.1
        Drop‑out probability after each hidden layer.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [8, 4]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.residual = nn.Linear(input_dim, 1) if input_dim!= 1 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.net(inputs)
        # residual connection from input to output
        return out + self.residual(inputs)


def EstimatorQNN() -> EstimatorNN:
    """Return an instance of the extended EstimatorNN."""
    return EstimatorNN()


__all__ = ["EstimatorQNN"]
