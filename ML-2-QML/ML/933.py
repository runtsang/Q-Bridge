"""Advanced feed‑forward regression network with batch normalization and dropout.

This module defines an `EstimatorQNN` class that extends the original
two‑layer architecture.  It now supports:
  * Two hidden layers with 16 and 8 units.
  * BatchNorm after each linear layer.
  * ReLU activations.
  * Dropout (p=0.2) for regularisation.
  * A forward method that returns the raw output.

The module can be used directly with PyTorch optimisers and loss
functions.
"""

import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """Regression network with two hidden layers, batch norm and dropout."""

    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int, int] = (16, 8),
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return network output."""
        return self.net(inputs)

__all__ = ["EstimatorQNN"]
