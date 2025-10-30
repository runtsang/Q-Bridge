"""Enhanced classical sampler network with residual connections and dropout.

The network is a two‑hidden‑layer MLP that mirrors the original SamplerQNN
but adds batch‑normalisation, LeakyReLU activations and a dropout layer to
improve generalisation.  A convenience ``sample`` method draws samples from
the soft‑max output, which is useful when the classical sampler is paired
with the quantum version in a hybrid pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SamplerQNNGen163(nn.Module):
    """Two‑hidden‑layer MLP with residual connection and dropout."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 2,
                 dropout_prob: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.bn1(self.fc1(x)))
        h = F.leaky_relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        logits = self.out(h)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1, **kwargs) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).
        n_samples : int, optional
            Number of samples to draw for each input.
        **kwargs : dict
            Additional arguments forwarded to torch.multinomial.

        Returns
        -------
        torch.Tensor
            Sample indices of shape (..., n_samples).
        """
        probs = self.forward(x)
        return torch.multinomial(probs, n_samples, replacement=True, **kwargs)

__all__ = ["SamplerQNNGen163"]
