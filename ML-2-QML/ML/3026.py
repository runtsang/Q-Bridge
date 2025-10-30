"""Hybrid classical kernel combining RBF and attention mechanisms."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class HybridKernelMethod(nn.Module):
    """
    A versatile kernel that fuses an RBF kernel with a self‑attention based
    similarity.  The same class can be dropped in place of the original
    ``Kernel`` module for classical machine‑learning workflows.
    """

    def __init__(self, gamma: float = 1.0, embed_dim: int = 4, alpha: float = 0.5):
        """
        Parameters
        ----------
        gamma : float
            Width parameter of the RBF kernel.
        embed_dim : int
            Dimensionality of the attention embedding.
        alpha : float
            Weight given to the attention component (0 = pure RBF,
            1 = pure attention).
        """
        super().__init__()
        self.gamma = gamma
        self.embed_dim = embed_dim
        self.alpha = alpha

        # Attention projections – simple linear layers for demonstration.
        self.q_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_lin = nn.Linear(embed_dim, embed_dim, bias=False)

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def _attention(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Query, key, and value projections.
        q = self.q_lin(x)
        k = self.k_lin(y)
        v = y
        # Scaled dot‑product attention.
        scores = torch.softmax((q @ k.T) / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ v
        # Map to a scalar similarity via inner product with a learnable vector.
        return torch.sum(attn_out * x, dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return a weighted sum of RBF and attention similarities.
        """
        x = x.view(1, -1)
        y = y.view(1, -1)
        rbf_sim = self._rbf(x, y)
        attn_sim = self._attention(x, y)
        return (1 - self.alpha) * rbf_sim + self.alpha * attn_sim

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of vectors.
        """
        return np.array([[self.forward(x, y).item() for y in b] for x in a])


__all__ = ["HybridKernelMethod"]
