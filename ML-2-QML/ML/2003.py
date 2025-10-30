"""Enhanced feed‑forward regressor with configurable depth, dropout and L2 regularisation.

This module retains the original EstimatorQNN signature – a callable that returns a
torch.nn.Module – while exposing a richer architecture.  The model can be instantiated
with arbitrary input dimensionality and a list of hidden layer sizes, making it suitable
for a wide range of regression tasks.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class _EstimatorQNN(nn.Module):
    """Configurable feed‑forward neural network."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        output_dim: int = 1,
        dropout: float = 0.0,
        l2_reg: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim

        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features

        layers.append(nn.Linear(in_features, output_dim))

        self.net = nn.Sequential(*layers)
        self.l2_reg = l2_reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

    def l2_loss(self) -> torch.Tensor:
        """Return the L2 regularisation term for all trainable weights."""
        if self.l2_reg == 0.0:
            return torch.tensor(0.0, device=self.net[0].weight.device)
        return self.l2_reg * sum(
            (p.pow(2).sum() for p in self.parameters() if p.requires_grad)
        )


def EstimatorQNN(
    input_dim: int = 2,
    hidden_layers: Sequence[int] | None = None,
    dropout: float = 0.0,
    l2_reg: float = 0.0,
) -> nn.Module:
    """Return a configurable regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input features.
    hidden_layers : Sequence[int] | None, default None
        Sizes of hidden layers.  If ``None`` a shallow 8→4 architecture is used.
    dropout : float, default 0.0
        Dropout probability applied after each hidden activation.
    l2_reg : float, default 0.0
        Coefficient for L2 weight decay.

    Returns
    -------
    nn.Module
        Instantiated neural network ready for training.
    """
    if hidden_layers is None:
        hidden_layers = [8, 4]
    return _EstimatorQNN(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=1,
        dropout=dropout,
        l2_reg=l2_reg,
    )


__all__ = ["EstimatorQNN"]
