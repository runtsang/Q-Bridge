"""Enhanced classical regressor with modular depth, regularisation and
dynamic input handling."""

from __future__ import annotations

import torch
from torch import nn


class EstimatorQNNEnhanced(nn.Module):
    """
    A fully‑connected regression network that supports:
    - Arbitrary input and output dimensionality
    - Multiple hidden layers with configurable widths
    - Batch‑normalisation and dropout for robust training
    - Easy extension for additional regularisers

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input feature vector.
    hidden_dims : list[int] or tuple[int], default (16, 8, 4)
        Widths of successive hidden layers.
    output_dim : int, default 1
        Dimensionality of the output regression vector.
    dropout : float, default 0.1
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int,...] = (16, 8, 4),
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_features = input_dim

        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_features = h

        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass through the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Regression output of shape ``(batch_size, output_dim)``.
        """
        return self.net(inputs)


__all__ = ["EstimatorQNNEnhanced"]
