"""
Enhanced classical sampler network.

This module defines a deep MLP with dropout and layer‑norm regularisation,
providing a `sample` convenience method that draws samples from the output
probabilities.  The public API mirrors the original seed by exposing a
`SamplerQNN()` function that returns an instance of the underlying class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class _SamplerQNN(nn.Module):
    """
    Deep, dropout‑regularised MLP for 2‑class probability estimation.

    Architecture:
        Input (2) → Linear(8) → LayerNorm → ReLU → Dropout(0.2)
        → Linear(4) → ReLU → Linear(2) → Softmax
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.LayerNorm(8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute class probabilities."""
        return F.softmax(self.net(inputs), dim=-1)

    def sample(self, num_samples: int, inputs: torch.Tensor) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the
        network's output probabilities.

        Parameters
        ----------
        num_samples:
            Number of samples to draw.
        inputs:
            Input batch of shape (batch_size, 2).

        Returns
        -------
        samples:
            Tensor of shape (batch_size, num_samples) containing class indices.
        """
        probs = self(inputs)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((num_samples,)).transpose(0, 1)


def SamplerQNN() -> _SamplerQNN:
    """
    Public factory that returns a ready‑to‑use sampler instance.
    """
    return _SamplerQNN()


__all__ = ["SamplerQNN"]
