"""Extended classical self‑attention module.

The class now exposes multi‑head attention, optional dropout and
a simple initialization routine.  It is fully PyTorch‑based and
compatible with existing transformer back‑ends.

Usage
-----
>>> sa = SelfAttention(embed_dim=64, num_heads=4, dropout=0.1)
>>> out = sa(inputs, rot, ent)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class SelfAttention(nn.Module):
    """Multi‑head self‑attention with optional dropout.

    Parameters
    ----------
    embed_dim : int
        Dimension of the input embeddings.
    num_heads : int, default 1
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        inputs : torch.Tensor, shape (batch, seq_len, embed_dim)
            Input embeddings.
        rotation_params : np.ndarray
            Parameters for the linear projections (unused but kept for API compatibility).
        entangle_params : np.ndarray
            Parameters for the linear projections (unused but kept for API compatibility).

        Returns
        -------
        torch.Tensor, shape (batch, seq_len, embed_dim)
            The attended representation.
        """
        batch, seq_len, _ = inputs.shape

        # Linear projections
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)

        # Reshape for multi‑head
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        context = torch.matmul(scores, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return context

    @staticmethod
    def init_params(
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        embed_dim: int,
        num_heads: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Utility to generate random parameters matching the API."""
        rng = np.random.default_rng()
        rot_shape = (embed_dim, embed_dim)
        ent_shape = (embed_dim, embed_dim)
        return rng.standard_normal(rot_shape), rng.standard_normal(ent_shape)

__all__ = ["SelfAttention"]
