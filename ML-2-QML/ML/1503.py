import torch
from torch import nn
from typing import Sequence

class EstimatorQNN(nn.Module):
    """
    A configurable feed‑forward regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input features.
    hidden_dims : Sequence[int], default (16, 8, 4)
        Sizes of successive hidden layers.
    dropout : float, default 0.1
        Dropout probability applied after each hidden layer.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: Sequence[int] = (16, 8, 4),
                 dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(dropout),
            ])
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., 1).
        """
        return self.net(x)

__all__ = ["EstimatorQNN"]
