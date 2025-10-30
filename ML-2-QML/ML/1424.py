"""Enhanced classical feed‑forward regressor.

The class supports arbitrary hidden layer sizes, batch‑normalisation,
drop‑out and a convenience loss method.  It retains the original
`EstimatorQNN` name so that downstream code can import it
unchanged."""
from __future__ import annotations

import torch
from torch import nn


class EstimatorQNN(nn.Module):
    """
    Configurable fully‑connected regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input features.
    hidden_dims : list[int], default [16, 8]
        Sizes of the hidden layers.
    dropout : float, default 0.1
        Drop‑out probability applied after each hidden block.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [16, 8]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the network output."""
        return self.net(x)

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean‑squared‑error loss."""
        return nn.functional.mse_loss(preds, targets)


__all__ = ["EstimatorQNN"]
