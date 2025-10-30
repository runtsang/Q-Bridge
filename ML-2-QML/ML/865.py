"""Enhanced feed‑forward regressor with configurable depth, dropout, and batch‑norm.

The original EstimatorQNN was a tiny 3‑layer network.  This version
provides a flexible architecture that can be tuned for a wide range
of regression tasks while still being lightweight enough for quick
experimentation.

Key features
------------
* Arbitrary hidden layer sizes via ``hidden_layers``.
* Optional dropout and batch‑normalisation per layer.
* Explicit ``activation`` argument for easy switching between
  Tanh, ReLU, Sigmoid, etc.
* A convenience factory function ``EstimatorQNN`` that returns an
  instance, preserving the API of the seed example.
"""

import torch
from torch import nn
from typing import Sequence

class EstimatorQNNModel(nn.Module):
    """Feed‑forward regression network with configurable depth."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: Sequence[int] = (8, 4),
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        # Resolve activation function
        act: nn.Module
        if activation.lower() == "tanh":
            act = nn.Tanh()
        elif activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "sigmoid":
            act = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def EstimatorQNN() -> EstimatorQNNModel:
    """Factory that mimics the original API and returns a ready‑to‑use model."""
    return EstimatorQNNModel()


__all__ = ["EstimatorQNN"]
