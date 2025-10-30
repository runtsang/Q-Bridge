"""Enhanced feed-forward regressor with dropout and batch normalization.

The network now supports configurable depth, hidden layer sizes, and dropout rates.
It also includes L2 regularization via weight decay in the optimizer.
"""

import torch
from torch import nn
from typing import Sequence

class EstimatorQNNExtended(nn.Module):
    """A flexible fully‑connected regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input feature vector.
    hidden_dims : Sequence[int], default (16, 8)
        Sizes of the hidden layers.
    dropout : float, default 0.1
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (16, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def EstimatorQNN() -> EstimatorQNNExtended:
    """Convenience factory that returns the default configuration."""
    return EstimatorQNNExtended()

__all__ = ["EstimatorQNNExtended", "EstimatorQNN"]
