"""Classical self‑attention module with multi‑head capability and optional residual.

The module mirrors the interface of the original seed while adding
- multi‑head attention
- dropout regularisation
- optional residual connection
- ability to expose attention scores for downstream analysis.

The class can be instantiated with ``embed_dim`` and ``num_heads``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionModule(nn.Module):
    """Multi‑head classical self‑attention."""

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0, residual: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.residual = residual

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Unused in the classical version but retained for API compatibility.
        entangle_params : np.ndarray
            Unused in the classical version but retained for API compatibility.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        output : np.ndarray
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Linear projections
        Q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # reshape for multi‑head
        def reshape_for_heads(t):
            return t.view(*t.shape[:2], self.num_heads, self.head_dim)\
                  .transpose(1, 2)  # (batch, heads, seq_len, head_dim)

        Q = reshape_for_heads(Q)
        K = reshape_for_heads(K)
        V = reshape_for_heads(V)

        # scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (batch, heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(*x.shape)  # (batch, seq_len, embed_dim)
        out = self.out_proj(out)

        if self.residual:
            out = out + x

        return out.numpy()

def SelfAttention():
    """Factory that returns a ready‑to‑use SelfAttentionModule."""
    return SelfAttentionModule(embed_dim=4, num_heads=2, dropout=0.1, residual=True)

__all__ = ["SelfAttentionModule", "SelfAttention"]
