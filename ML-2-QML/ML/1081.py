"""Enhanced feed‑forward regressor with residual connections and dropout.

The original EstimatorQNN was a tiny 3‑layer network.  This extension adds
* residual skip connections between every two layers
* optional dropout for regularisation
* a configurable hidden size and depth
* a small utility to compute the mean‑squared error on a batch.

The public API is unchanged: EstimatorQNNEnhanced() returns an nn.Module
instance that can be trained with any standard PyTorch optimiser.
"""

from __future__ import annotations

import torch
from torch import nn


class EstimatorQNNEnhanced(nn.Module):
    """Residual‑enabled feed‑forward regressor."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for _ in range(depth):
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        self.out = nn.Linear(hidden_dim, 1)
        self.residuals = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = x + self.residuals[i](residual)
        return self.out(x)

    @staticmethod
    def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return mean‑squared error."""
        return torch.mean((pred - target) ** 2)


__all__ = ["EstimatorQNNEnhanced"]
