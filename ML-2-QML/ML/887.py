"""
Classical deep sampler network with configurable depth and dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A configurable neural sampler.

    Parameters
    ----------
    input_dim : int
        Size of the input feature vector.
    hidden_dims : list[int]
        List of hidden layer sizes.
    output_dim : int
        Number of output classes (default 2).
    dropout : float
        Dropout probability applied after each hidden layer.

    Notes
    -----
    The network ends with a softmax to produce a categorical distribution.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | None = None,
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [4]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing class probabilities.
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).
        n_samples : int
            Number of samples to draw per input.

        Returns
        -------
        torch.Tensor
            Sampled indices of shape (..., n_samples).
        """
        probs = self.forward(x)
        return torch.multinomial(probs, n_samples, replacement=True)


__all__ = ["SamplerQNN"]
