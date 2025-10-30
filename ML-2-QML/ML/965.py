"""
Enhanced classical feed‑forward regressor.

Implements a deeper network with batch‑normalisation, dropout and a larger hidden
representation.  The architecture is deliberately over‑parameterised to provide
a richer feature extraction capability while still remaining lightweight for
quick experimentation.

The class is designed to be drop‑in compatible with the original EstimatorQNN
factory: it can be instantiated directly and used as a `torch.nn.Module`.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class EstimatorQNNEnhanced(nn.Module):
    """
    A fully‑connected regression network with:
      * 2 input features
      * 3 hidden layers (16 → 32 → 16 units)
      * Batch‑normalisation after each linear layer
      * ReLU activations and 30 % dropout
      * 1‑dimensional output

    The model is intentionally deeper than the seed version to allow the
    exploration of more complex function approximations while keeping the
    parameter count modest.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch_size, 2).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch_size, 1).
        """
        return self.net(inputs)

    def predict(self, inputs: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
        """
        Convenience method for inference.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.
        device : str | None
            Target device. If ``None``, the model's device is used.

        Returns
        -------
        torch.Tensor
            Predicted values.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(inputs.to(device))

__all__ = ["EstimatorQNNEnhanced"]
