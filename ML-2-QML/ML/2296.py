"""Hybrid kernel combining RBF similarity and self‑attention weighting.

This module builds on the classical RBF kernel and classical self‑attention
implementations to produce a richer similarity measure.  The kernel value
between two samples is the product of an RBF term and an attention score
computed from a lightweight feed‑forward network.  The design mirrors the
quantum side where a quantum kernel is multiplied by a quantum‑derived
attention weight.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence

class HybridKernelAttention(nn.Module):
    """
    Hybrid RBF‑attention kernel.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input samples.
    embed_dim : int, default 4
        Dimensionality of the attention embedding.
    gamma : float, default 1.0
        RBF bandwidth.
    init_scale : float, default 0.1
        Scale for initializing attention parameters.
    """

    def __init__(self,
                 input_dim: int,
                 embed_dim: int = 4,
                 gamma: float = 1.0,
                 init_scale: float = 0.1) -> None:
        super().__init__()
        self.gamma = gamma
        self.embed_dim = embed_dim

        # Linear projection to the attention space
        self.embed = nn.Linear(input_dim, embed_dim, bias=False)

        # Parameters that emulate the rotation and entanglement
        # of the quantum self‑attention block
        self.rotation_params = nn.Parameter(
            torch.randn(embed_dim * 3) * init_scale)
        self.entangle_params = nn.Parameter(
            torch.randn(embed_dim - 1) * init_scale)

    def _attention_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute a scalar attention weight between two embedded vectors.
        """
        # Embed inputs
        x_e = self.embed(x)
        y_e = self.embed(y)

        # Compute query, key
        query = torch.mm(x_e, self.rotation_params.reshape(self.embed_dim, -1))
        key = torch.mm(y_e, self.entangle_params.reshape(self.embed_dim - 1, -1))

        # Attention scores
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        # Use the (0,1) entry as the attention from x to y
        return scores[0, 1].unsqueeze(0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return the hybrid kernel value for a single pair.
        """
        # RBF part
        diff = x - y
        rbf = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

        # Attention weight
        attn = self._attention_score(x, y)

        return rbf * attn

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """
    Compute the Gram matrix for two collections of samples using the
    hybrid RBF‑attention kernel.
    """
    kernel = HybridKernelAttention(input_dim=a[0].shape[-1], gamma=gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernelAttention", "kernel_matrix"]
