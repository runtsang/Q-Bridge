"""
EnhancedSamplerQNN: A flexible classical sampler network.

Features
--------
* Configurable input dimension and hidden layer sizes.
* Optional dropout for regularisation.
* Dual output modes: softmax probabilities or log‑softmax values.
* Sampling interface that draws from the categorical distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class EnhancedSamplerQNN(nn.Module):
    """
    A two‑layer (extendable) neural sampler with optional dropout.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input tensor.
    hidden_dims : list[int], default [8, 4]
        Sizes of hidden layers. The last element becomes the output dimension.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 4]
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        # Final layer maps to the number of output classes
        layers.append(nn.Linear(in_dim, hidden_dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return softmax probabilities over the output classes.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (..., output_dim).
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return log‑softmax values for use in likelihood calculations.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Log‑softmax of shape (..., output_dim).
        """
        logits = self.net(x)
        return F.log_softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw categorical samples from the predicted distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        num_samples : int, default 1
            Number of samples to draw per input.

        Returns
        -------
        torch.Tensor
            Sample indices of shape (num_samples, *x.shape[:-1]).
        """
        probs = self.forward(x)
        dist = Categorical(probs)
        return dist.sample((num_samples,)).transpose(0, 1)

__all__ = ["EnhancedSamplerQNN"]
