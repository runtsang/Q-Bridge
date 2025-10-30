"""Extended feed‑forward regressor with residual connections.

Features
--------
* Residual connections for better gradient flow
* Batch‑norm and dropout for regularisation
* Configurable hidden sizes via ``hidden_sizes`` argument
"""

import torch
from torch import nn


class EstimatorQNN(nn.Module):
    """
    A robust regression network that extends the original 2‑layer MLP.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_sizes: tuple[int,...] | list[int] = (16, 32, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = size

        # Residual branch
        if input_dim == hidden_sizes[0]:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(input_dim, hidden_sizes[0])

        self.main = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual addition.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., 1) containing the regression output.
        """
        residual = self.residual(x)
        out = self.main(x)
        out = out + residual
        out = self.out(out)
        return out


__all__ = ["EstimatorQNN"]
