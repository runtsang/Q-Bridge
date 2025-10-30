"""Enhanced feed-forward regressor with residual connections and dropout.

This module defines EstimatorQNN as a factory that returns a PyTorch
model capable of learning more complex functions than the original 2‑layer
network.  The architecture consists of multiple hidden layers, ReLU
activations, dropout for regularisation and a residual block that adds
the input to the output of the first hidden layer.  The design is still
fully compatible with the original API: calling EstimatorQNN() returns
an nn.Module instance.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block that adds its input to a linear transform."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(x)


class EstimatorNN(nn.Module):
    """Deep residual feed‑forward regressor."""
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | tuple[int,...] = (16, 8),
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim

        # First hidden layer with a residual connection
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(ResidualBlock(hidden_dims[0]))
        prev_dim = hidden_dims[0]

        # Remaining hidden layers
        for h in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def EstimatorQNN() -> nn.Module:
    """Return an instance of the enhanced estimator."""
    return EstimatorNN()
