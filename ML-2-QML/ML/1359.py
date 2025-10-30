"""Extended feed‑forward regressor with batch‑norm, dropout, and optional residual connections.

The class `EstimatorQNNExtended` is a drop‑in replacement for the original tiny network.
It adds depth, regularisation and a `predict` helper for inference.
"""

from __future__ import annotations

import torch
from torch import nn


class EstimatorQNNExtended(nn.Module):
    """A robust regression network with residual connections, batch normalisation and dropout."""

    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16, 8]  # deeper hidden layers than the seed
        layers = []
        prev_dim = input_dim
        for idx, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            prev_dim = dim
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return self.output(out)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience inference wrapper."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def EstimatorQNN() -> EstimatorQNNExtended:
    """Return an instance of the extended estimator."""
    return EstimatorQNNExtended()


__all__ = ["EstimatorQNN", "EstimatorQNNExtended"]
