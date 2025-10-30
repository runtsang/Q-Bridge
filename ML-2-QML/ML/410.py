"""
Robust classical self‑attention module with multi‑head, dropout, and
gradient support.

The module accepts a 3‑D tensor of shape (batch, seq_len, embed_dim) and
produces a transformed representation.  The interface mirrors the original
seed: a ``run`` method that forwards to ``forward`` for use in pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ClassicalSelfAttention(nn.Module):
    """
    Multi‑head self‑attention with dropout and optional bias.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, value
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=True)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=True)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=True)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        out : torch.Tensor
            Attention‑weighted sum of values, shape (batch, seq_len, embed_dim).
        attn_weights : torch.Tensor
            Normalized attention weights, shape (batch, num_heads, seq_len, seq_len).
        """
        batch, seq_len, _ = x.size()

        # Linear projections
        Q = self.Wq(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return out, attn_weights

    def run(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compatibility wrapper that ignores the quantum‑style parameters
        and forwards to :meth:`forward`.

        Parameters
        ----------
        rotation_params, entangle_params : torch.Tensor
            Unused; kept for API compatibility.
        inputs : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention output of shape (batch, seq_len, embed_dim).
        """
        out, _ = self.forward(inputs)
        return out

__all__ = ["ClassicalSelfAttention"]
