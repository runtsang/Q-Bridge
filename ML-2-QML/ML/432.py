"""
EstimatorQNN (classical)

A versatile regression network that extends the original tiny model
by adding residual connections, dropout, and an automatically inferred
input dimension.  It can be used wherever a PyTorch `nn.Module` is
expected.
"""

import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """
    Robust feed‑forward regressor.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input features.
    hidden_dims : tuple[int,...], default (32, 16)
        Sizes of successive hidden layers.
    dropout : float, default 0.1
        Dropout probability applied after every hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int,...] = (32, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch_size, 1).
        """
        return self.net(x)

__all__ = ["EstimatorQNN"]
