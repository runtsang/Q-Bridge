"""
HybridSamplerAttention – classical two‑stage network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridSamplerAttention(nn.Module):
    """
    Classical hybrid network that first applies a small feed‑forward sampler
    and then a self‑attention mechanism.  Parameters are fully trainable
    and can be optimised jointly with any downstream loss.
    """

    def __init__(self, embed_dim: int = 4, sampler_hidden: int = 4) -> None:
        """
        Parameters
        ----------
        embed_dim: int
            Dimensionality of the attention feature space.
        sampler_hidden: int
            Number of hidden units in the sampler.
        """
        super().__init__()
        # Sampler stage – mirrors the original linear chain
        self.sampler = nn.Sequential(
            nn.Linear(2, sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, embed_dim),
        )
        # Attention stage – similar to the classical SelfAttention
        self.embed_dim = embed_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through sampler followed by attention.

        Parameters
        ----------
        inputs: torch.Tensor
            Shape (..., 2) – batch of 2‑dimensional input vectors.

        Returns
        -------
        torch.Tensor
            Output of the attention block, shape (..., embed_dim).
        """
        # Sampler output
        sampler_out = self.sampler(inputs)  # (..., embed_dim)
        # Classic attention using the sampler output as query/key/value
        query = sampler_out
        key = sampler_out
        value = inputs
        scores = F.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).squeeze(-1)

__all__ = ["HybridSamplerAttention"]
