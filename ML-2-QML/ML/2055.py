"""
Fully connected layer with trainable parameters and optional regularisation.
The class can be dropped into larger PyTorch models and supports standard
optimisation hooks.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class FCL(nn.Module):
    """
    A simple, fully‑connected layer.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int, default 1
        Number of output neurons.
    dropout : float, optional
        Dropout probability. If None, dropout is disabled.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., output_dim).
        """
        out = self.linear(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return self.activation(out)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Compatibility method that mirrors the seed API.
        Converts a sequence of scalars into a tensor, runs them through
        the layer, and returns the mean of the output.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of parameters to feed into the linear layer.

        Returns
        -------
        torch.Tensor
            1‑D tensor containing the mean output.
        """
        values = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        output = self.forward(values)
        return output.mean(dim=0)

__all__ = ["FCL"]
