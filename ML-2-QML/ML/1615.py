"""
Standard PyTorch regression network with configurable depth, dropout
and batch‑normalisation.  The design is deliberately modular so that
users can experiment with depth, width and regularisation without
changing the training loop.
"""

from __future__ import annotations

import torch
from torch import nn


class EstimatorQNN(nn.Module):
    """
    A fully‑connected regression network that supports:

    * variable hidden width (`hidden_size`)
    * variable depth (`num_layers`)
    * dropout (`p_dropout`)
    * batch‑normalisation (if requested)
    * optional skip connections between every other layer

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_size : int, default=64
        Number of units in each hidden layer.
    num_layers : int, default=4
        Number of hidden layers.
    p_dropout : float, default=0.1
        Dropout probability.
    use_batchnorm : bool, default=False
        Whether to intersperse `nn.BatchNorm1d` layers.
    skip_connections : bool, default=False
        Adds a residual connection every two layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 4,
        p_dropout: float = 0.1,
        use_batchnorm: bool = False,
        skip_connections: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p_dropout))
            in_dim = hidden_size

        # Optional skip connections (every two layers)
        if skip_connections and num_layers >= 2:
            self.skip_indices = list(range(0, num_layers, 2))
        else:
            self.skip_indices = []

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional residual connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch_size, 1).
        """
        residual = x
        for idx, layer in enumerate(self.hidden):
            x = layer(x)
            # Inject skip connection after every two layers
            if idx in self.skip_indices:
                x = x + residual
        return self.output(x)


def EstimatorQNN_factory(**kwargs) -> EstimatorQNN:
    """
    Convenience factory that returns a configured EstimatorQNN.

    Example
    -------
    >>> net = EstimatorQNN_factory(input_dim=2, hidden_size=128, num_layers=6,
   ...                            p_dropout=0.2, use_batchnorm=True,
   ...                            skip_connections=True)
    """
    return EstimatorQNN(**kwargs)


__all__ = ["EstimatorQNN", "EstimatorQNN_factory"]
