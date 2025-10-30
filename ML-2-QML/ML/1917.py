"""Extended classical estimator with configurable architecture and utilities.

This module builds on the original EstimatorQNN by allowing arbitrary hidden
layer sizes, optional dropout and batch‑norm, and a prediction helper that
disables gradients automatically.  The returned class is named
`EstimatorQNNExtended` and can be instantiated directly or via the
`create_estimator_qnn_extended` factory for clearer intent.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence

class EstimatorQNNExtended(nn.Module):
    """
    A flexible fully‑connected regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input features.
    hidden_dims : Sequence[int], default (16, 8)
        Sizes of hidden layers.
    dropout : float, default 0.0
        Dropout probability; set to 0 to disable.
    batch_norm : bool, default False
        Whether to insert BatchNorm1d after each hidden Linear layer.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (16, 8),
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions in evaluation mode without gradients."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

def create_estimator_qnn_extended(**kwargs) -> EstimatorQNNExtended:
    """
    Convenience factory that forwards kwargs to EstimatorQNNExtended.
    """
    return EstimatorQNNExtended(**kwargs)

__all__ = ["EstimatorQNNExtended", "create_estimator_qnn_extended"]
