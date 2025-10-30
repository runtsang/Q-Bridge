"""Enhanced classical sampler network with configurable depth and dropout."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


class SamplerQNN(nn.Module):
    """
    A flexible neural network sampler that accepts any input dimensionality
    and supports multiple hidden layers with dropout for regularisation.

    Parameters
    ----------
    input_dim : int
        Size of the input feature vector.
    hidden_dims : Sequence[int], optional
        Sizes of intermediate hidden layers. Defaults to (16, 8).
    output_dim : int
        Number of output probability classes. Defaults to 2.
    dropout_rate : float, optional
        Dropout probability applied after each hidden layer. Defaults to 0.1.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (16, 8),
        output_dim: int = 2,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass computing a probability distribution via softmax.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (..., output_dim).
        """
        return F.softmax(self.net(inputs), dim=-1)


__all__ = ["SamplerQNN"]
