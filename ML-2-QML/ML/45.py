"""Enhanced feed‑forward regressor with residual connections and dropout.

This module defines :class:`EstimatorQNN`, a PyTorch neural network that
accepts an arbitrary input dimensionality, stacks several hidden layers
with batch‑norm, ReLU and dropout, and finally projects to a single
regression output.  A residual skip connection is added between the
first and last hidden layers to ease optimisation.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence, Tuple

class EstimatorQNN(nn.Module):
    """Residual feed‑forward regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input feature vector.
    hidden_dims : Sequence[int], default (8, 4)
        Sizes of the hidden layers.  The first and last hidden layers
        are connected by a skip connection.
    dropout_rate : float, default 0.1
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (8, 4),
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout_rate = dropout_rate

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        self.features = nn.Sequential(*layers)

        # Linear layer to produce the final output
        self.output_layer = nn.Linear(self.hidden_dims[-1], 1)

        # Residual projection if dimensions differ
        if input_dim!= self.hidden_dims[0]:
            self.residual = nn.Linear(input_dim, self.hidden_dims[-1])
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, input_dim)``.
        """
        h = self.features(x)
        # Residual skip
        res = self.residual(x)
        h = h + res
        out = self.output_layer(h)
        return out
