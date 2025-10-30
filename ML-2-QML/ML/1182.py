"""Enhanced classical estimator with configurable depth, dropout and optional batch normalization."""

import torch
from torch import nn

class EstimatorQNNExtended(nn.Module):
    """
    A flexible fully‑connected regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input features.
    hidden_dims : list[int] | tuple[int,...], default (8, 8)
        Sizes of hidden layers.
    dropout : float, default 0.1
        Dropout probability applied after each hidden layer.
    use_bn : bool, default False
        Whether to insert BatchNorm1d after each hidden linear layer.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: list[int] | tuple[int,...] = (8, 8),
                 dropout: float = 0.1,
                 use_bn: bool = False) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1).
        """
        return self.net(x)

def EstimatorQNN() -> EstimatorQNNExtended:
    """
    Helper that returns the default EstimatorQNNExtended instance
    compatible with the original EstimatorQNN interface.
    """
    return EstimatorQNNExtended()

__all__ = ["EstimatorQNNExtended", "EstimatorQNN"]
