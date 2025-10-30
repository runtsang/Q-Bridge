"""Enhanced classical regressor with configurable depth and regularisation."""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence

class EstimatorQNNHybrid(nn.Module):
    """
    A flexible fully‑connected regression network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_layers : Sequence[int], optional
        Sizes of hidden layers. Defaults to ``[64, 32]``.
    dropout : float, optional
        Drop‑out probability applied after each hidden layer.
    batch_norm : bool, optional
        Whether to insert a BatchNorm1d layer after each linear layer.
    activation : nn.Module, optional
        Activation function to use. Defaults to nn.Tanh.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: Sequence[int] | None = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: nn.Module = nn.Tanh,
    ) -> None:
        super().__init__()
        hidden_layers = hidden_layers or [64, 32]
        layers: list[nn.Module] = []

        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, input_dim)``.
        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, 1)``.
        """
        return self.net(x)

    @staticmethod
    def default() -> "EstimatorQNNHybrid":
        """Convenience constructor with the default architecture."""
        return EstimatorQNNHybrid()

__all__ = ["EstimatorQNNHybrid"]
