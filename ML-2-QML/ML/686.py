"""EstimatorQNN – enhanced classical regression model.

This module defines a flexible, deep feed‑forward network that can be
used for regression tasks.  The architecture supports an arbitrary
sequence of hidden layers, layer‑normalisation, ReLU activations,
dropout and a final linear head.  The design mirrors the original
`EstimatorQNN` while adding depth and regularisation, making it
suitable for more complex datasets.

Typical usage:

>>> from EstimatorQNN__gen350 import EstimatorQNN
>>> model = EstimatorQNN(input_dim=2, hidden_dims=[64, 32], dropout=0.2)
>>> x = torch.randn(5, 2)
>>> y = model(x)
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence


class EstimatorQNN(nn.Module):
    """
    A configurable deep regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input features.
    hidden_dims : Sequence[int], default (64, 32)
        Sizes of successive hidden layers.
    output_dim : int, default 1
        Dimensionality of the output.
    dropout : float, default 0.1
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (64, 32),
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        return self.net(x)


__all__ = ["EstimatorQNN"]
