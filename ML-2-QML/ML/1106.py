"""
AdvancedSamplerQNN: A classical neural sampler with dropout and categorical sampling.

This module extends the original two‑layer network by adding a hidden layer,
dropout for regularisation, and a convenient `sample` method that draws
samples from the learned categorical distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AdvancedSamplerQNN(nn.Module):
    """
    A two‑input, two‑output neural sampler with a hidden layer and dropout.
    The network outputs a probability distribution over two classes.
    """

    def __init__(self, hidden_dim: int = 8, dropout_rate: float = 0.2) -> None:
        """
        Parameters
        ----------
        hidden_dim : int
            Number of units in the hidden layer.
        dropout_rate : float
            Dropout probability applied after the hidden activation.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing class probabilities.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (..., 2) representing two input features.

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (..., 2).
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw categorical samples from the network's output distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., 2).
        n_samples : int
            Number of samples to draw for each input.

        Returns
        -------
        torch.Tensor
            Integer samples of shape (..., n_samples).
        """
        probs = self.forward(inputs)
        # Use Gumbel‑softmax trick for differentiable sampling if needed
        return torch.multinomial(probs, n_samples, replacement=True)

__all__ = ["AdvancedSamplerQNN"]
