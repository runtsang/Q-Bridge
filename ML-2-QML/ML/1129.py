"""SamplerQNNGen – a classical sampler with dropout and batch‑norm.

The new architecture extends the original 2‑layer network with a
three‑layer MLP, dropout for regularisation and a batch‑norm layer.
A convenient `sample` method is added to draw categorical samples
directly from the output probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class SamplerModule(nn.Module):
    """Deep MLP with dropout and batch‑norm for probability estimation."""

    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int, int] = (8, 8), output_dim: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class probabilities with softmax."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Draw categorical samples from the output distribution."""
        probs = self.forward(inputs)
        dist = Categorical(probs)
        return dist.sample((num_samples,)).squeeze(0)


def SamplerQNN() -> SamplerModule:
    """Factory that returns the extended sampler module."""
    return SamplerModule()


__all__ = ["SamplerQNN"]
