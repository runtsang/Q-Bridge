"""Enhanced classical regressor for EstimatorQNN.

The network is now a configurable feed‑forward model with
batch‑normalisation, ReLU non‑linearity, and dropout.  It accepts an
arbitrary number of hidden layers and can be tuned via the constructor.
"""

from __future__ import annotations

import torch
from torch import nn


class EstimatorNN(nn.Module):
    """A flexible fully‑connected regression network."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | tuple[int,...] | None = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        # Build hidden layers
        for h in hidden_dims or []:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


def EstimatorQNN() -> EstimatorNN:
    """Return a ready‑to‑use EstimatorNN instance."""
    return EstimatorNN()


__all__ = ["EstimatorQNN"]
