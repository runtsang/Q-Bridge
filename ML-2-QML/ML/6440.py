"""Hybrid classical kernel + self‑attention module."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

class HybridQuantumKernelAttention(nn.Module):
    """Hybrid kernel‑attention module mimicking the quantum interface.

    The module first computes an RBF kernel matrix between the input ``x`` and a
    reference set ``ref`` and then applies a classical self‑attention mechanism
    over the kernel representation.  It is fully compatible with the quantum
    counterpart defined in ``QuantumKernelMethod__gen262.py`` but remains
    purely classical.
    """

    def __init__(self, gamma: float = 1.0, embed_dim: int = 4) -> None:
        super().__init__()
        self.gamma = gamma
        self.embed_dim = embed_dim

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel matrix."""
        a = a.unsqueeze(1)  # (n, 1, d)
        b = b.unsqueeze(0)  # (1, m, d)
        diff = a - b
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))

    def attention(self, kernel: torch.Tensor) -> torch.Tensor:
        """Classical self‑attention over the kernel matrix."""
        # Treat kernel as embeddings: query, key, value = kernel
        query = kernel
        key = kernel
        value = kernel
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Compute the hybrid representation for ``x`` given reference set ``ref``."""
        k = self.kernel_matrix(x, ref)
        return self.attention(k)

__all__ = ["HybridQuantumKernelAttention"]
