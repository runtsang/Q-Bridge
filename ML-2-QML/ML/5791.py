"""EstimatorQNNExtended – a richer classical regressor.

The network builds upon the original two‑layer network by adding:
* Residual connections to ease optimisation.
* Batch‑normalisation layers for stable gradients.
* Dropout for regularisation.
* A flexible forward method that can be used directly in PyTorch training loops.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class EstimatorQNNExtended(nn.Module):
    """
    A fully‑connected regression network with residual connections, batch‑norm
    and dropout.  Designed to be plug‑in compatible with the original
    EstimatorQNN example while providing a more expressive architecture.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Optional[list[int]] = None,
        output_dim: int = 1,
        dropout_prob: float = 0.2,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of input features.
        hidden_dims : list[int] | None
            Sequence of hidden layer sizes.  Defaults to [16, 32, 16].
        output_dim : int
            Dimensionality of the output (typically 1 for regression).
        dropout_prob : float
            Dropout probability applied after every hidden layer.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 32, 16]

        layers = []
        prev_dim = input_dim
        self.residuals = nn.ModuleList()

        for idx, dim in enumerate(hidden_dims):
            # Linear + BatchNorm + Activation
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_prob))

            # Residual shortcut if dimensions match
            if prev_dim == dim:
                self.residuals.append(nn.Identity())
            else:
                self.residuals.append(nn.Linear(prev_dim, dim))

            prev_dim = dim

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        out = x
        for idx, layer in enumerate(self.hidden):
            # Each block consists of 4 sub‑layers: Linear, BN, ReLU, Dropout
            out = layer(out)
            # Add residual after the block
            if idx % 4 == 3:
                residual = self.residuals[idx // 4](x)
                out = out + residual
                x = out  # update shortcut input for next block

        return self.output_layer(out)


__all__ = ["EstimatorQNNExtended"]
