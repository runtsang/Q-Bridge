"""Enhanced feed‑forward regression model with dropout, batch normalisation and optional residual connections.

The function :func:`EstimatorQNN` returns an instance of :class:`EstimatorNN` which is a drop‑in
replacement for the original 2‑layer network.  The architecture still accepts two‑dimensional
inputs and produces a single scalar output, but it now contains

* two hidden layers of size 8 and 4 respectively,
* ``nn.BatchNorm1d`` after each linear layer,
* ``nn.Dropout`` with a configurable probability,
* a residual connection from the input to the first hidden layer.

The design demonstrates how a lightweight model can be extended without changing the API
used by downstream code.  The network can be trained with standard PyTorch optimisers.
"""

from __future__ import annotations

import torch
from torch import nn


class EstimatorNN(nn.Module):
    """
    Regression network that mirrors the original EstimatorQNN but with improved
    regularisation and a residual skip connection.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int, int] = (8, 4),
        dropout: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        input_dim:
            Dimensionality of the input vector.
        hidden_dims:
            Sizes of the two hidden layers.
        dropout:
            Dropout probability applied after every hidden layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Linear layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = nn.Linear(input_dim, hidden_dims[0])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        The input is projected to the first hidden layer both directly and via a
        residual connection.  The two representations are summed before activation.
        """
        # Residual path
        resid = self.residual(inputs)

        # Main path
        h1 = self.fc1(inputs)
        h1 = self.bn1(h1)
        h1 = torch.tanh(h1)

        # Add residual
        h1 = h1 + resid
        h1 = self.dropout(h1)

        h2 = self.fc2(h1)
        h2 = self.bn2(h2)
        h2 = torch.tanh(h2)
        h2 = self.dropout(h2)

        out = self.fc3(h2)
        return out


def EstimatorQNN() -> EstimatorNN:
    """Return an instance of the extended EstimatorNN."""
    return EstimatorNN()


__all__ = ["EstimatorQNN"]
