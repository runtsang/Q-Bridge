"""Classical sampler network with extended architecture and regularization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNNGen089(nn.Module):
    """
    A flexible sampler network that mirrors the original SamplerQNN but adds
    support for arbitrary input/output dimensions, hidden layers, dropout,
    and batch normalization.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input vector.
    output_dim : int, default 2
        Number of output classes.
    hidden_dims : list[int] or tuple[int,...], default (4,)
        Sizes of hidden layers.
    dropout : float, default 0.1
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        hidden_dims: list[int] | tuple[int,...] = (4,),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing class probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (batch_size, output_dim).
        """
        logits = self.network(x)
        return F.softmax(logits, dim=-1)


__all__ = ["SamplerQNNGen089"]
