"""
EstimatorQNN__gen153 – Classical deep‑learning regressor with configurable depth.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence, Optional


class EstimatorQNN__gen153(nn.Module):
    """
    A flexible fully‑connected regressor.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    hidden_layers : Sequence[int]
        Sizes of hidden layers.  Defaults to (16, 8).
    dropout : float, optional
        Dropout probability applied after every hidden layer.  ``None`` disables dropout.
    batch_norm : bool, default=False
        Whether to insert a BatchNorm1d layer after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: Sequence[int] = (16, 8),
        dropout: Optional[float] = None,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        prev_dim = input_dim
        for idx, h_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Regression output of shape ``(batch_size, 1)``.
        """
        return self.net(inputs)


def EstimatorQNN__gen153() -> EstimatorQNN__gen153:
    """
    Factory that returns a ready‑to‑train instance with default hyper‑parameters.
    """
    return EstimatorQNN__gen153()


__all__ = ["EstimatorQNN__gen153"]
