"""Enhanced multi‑head self‑attention module with dropout and residual connections.

The class mirrors the original API but adds a full transformer‑style
attention block that can be dropped into larger models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi‑head self‑attention layer.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads. Must divide embed_dim.
    dropout : float
        Dropout probability applied to attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        mask : torch.Tensor | None
            Optional attention mask of shape (batch, seq_len, seq_len).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = x.size()

        # Linear projections
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        # Residual connection + output projection
        return self.out_proj(context + x)


def SelfAttention(embed_dim: int = 4, num_heads: int = 4, dropout: float = 0.1) -> MultiHeadSelfAttention:
    """
    Factory that returns a ready‑to‑use multi‑head self‑attention module.

    The signature matches the original ``SelfAttention`` function while
    exposing additional transformer‑style options.
    """
    return MultiHeadSelfAttention(embed_dim, num_heads, dropout)


__all__ = ["SelfAttention", "MultiHeadSelfAttention"]
