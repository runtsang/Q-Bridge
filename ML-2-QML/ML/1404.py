"""Enhanced MLP regressor with configurable architecture and regularisation.

The original EstimatorQNN was a tiny two‑layer network.  This version
exposes a flexible interface that can be used for quick prototyping
or as a drop‑in replacement in larger pipelines.  It supports:
- arbitrary input dimensionality
- a stack of hidden layers with ReLU activations
- optional batch‑normalisation and dropout
- a final linear output for regression
"""

import torch
from torch import nn
from typing import Sequence

class EstimatorNN(nn.Module):
    """A configurable feed‑forward network for regression."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 4]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

def EstimatorQNN() -> EstimatorNN:
    """Return an instance of the default regressor."""
    return EstimatorNN()

__all__ = ["EstimatorQNN"]
