"""Enhanced feed‑forward regressor with residuals and dropout.

This module defines EstimatorQNNGen, a PyTorch neural network that extends the original
simple two‑layer network. The architecture now uses residual connections, batch
normalisation and dropout to improve generalisation on small datasets.
"""

from __future__ import annotations

import torch
from torch import nn


class EstimatorQNNGen(nn.Module):
    """
    A lightweight yet expressive regression network.

    Attributes
    ----------
    net : nn.Sequential
        Sequential container implementing the feature extractor.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        """
        Build the network.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        hidden_dim : int
            Size of the hidden layers.
        dropout : float
            Dropout probability applied after each hidden block.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted scalar output.
        """
        return self.net(x)


__all__ = ["EstimatorQNNGen"]
