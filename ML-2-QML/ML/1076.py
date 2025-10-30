"""Enhanced classical regression estimator.

This module defines EstimatorQNN, a deeper feed‑forward neural network
with batch‑normalisation and dropout.  It mirrors the original
EstimatorQNN interface but offers improved capacity for learning
non‑linear relationships in small tabular datasets.
"""

from __future__ import annotations

import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """Deep regression network with regularisation.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input features.
    hidden_dims : list[int], default [16, 16, 8]
        Sizes of the hidden layers.
    dropout : float, default 0.1
        Drop‑out probability applied after each hidden layer.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [16, 16, 8]
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass.

        Parameters
        ----------
        inputs
            Tensor of shape ``(batch_size, input_dim)``.
        """
        return self.net(inputs)

__all__ = ["EstimatorQNN"]
