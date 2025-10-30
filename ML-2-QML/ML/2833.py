"""Hybrid classical sampler with self‑attention.

This module defines SamplerQNNGen094, a torch.nn.Module that first applies a
classical self‑attention block to the input, then feeds the attended
features into a lightweight feed‑forward sampler network.  The design
mirrors the original SamplerQNN but adds an attention layer, improving
expressivity while keeping the model fully classical and
compatible with PyTorch training pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SamplerQNNGen094(nn.Module):
    """
    Hybrid sampler network: self‑attention → feed‑forward sampler.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of the attention space. Default 4.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Self‑attention parameters
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        # Sampler network
        self.sampler_net = nn.Sequential(
            nn.Linear(embed_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def _attention(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Classic scaled dot‑product attention using the learned rotation and
        entangle parameters.
        """
        query = inputs @ self.rotation_params
        key = inputs @ self.entangle_params
        value = inputs
        scores = F.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor of shape (batch, feature)
            Input feature vector.

        Returns
        -------
        torch.Tensor
            Softmaxed probability distribution of shape (batch, 2).
        """
        attended = self._attention(inputs)
        logits = self.sampler_net(attended)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNNGen094"]
