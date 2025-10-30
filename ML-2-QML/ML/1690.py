"""
Extended multi‑head self‑attention module compatible with the classic Transformer block.
The interface mirrors the original seed so that existing pipelines can switch to the new
implementation without code changes.  Optional parameters `rotation_params` and
`entangle_params` are accepted for API parity but ignored by the classical version.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention module.

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
        self.scale = self.head_dim ** -0.5

        # Linear projections for query, key, value
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: torch.Tensor = None,
        entangle_params: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params, entangle_params : torch.Tensor, optional
            Parameters kept for API compatibility with the quantum version.
            They are ignored.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        attn_scores = torch.einsum("bnhd,bmhd->bhnm", q, k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum("bhnm,bmhd->bnhd", attn_weights, v)
        attn_output = attn_output.reshape(B, N, self.embed_dim)
        return self.out_proj(attn_output)


def SelfAttention():
    """
    Factory that preserves the original seed's callable interface.
    Returns a SelfAttention instance configured for 4‑dimensional embeddings.
    """
    return SelfAttention(embed_dim=4)


__all__ = ["SelfAttention"]
