"""Enhanced classical self‑attention with multi‑head support and dropout.

This module defines a SelfAttention class that mirrors the interface of the
quantum self‑attention circuit.  The class implements a multi‑head
self‑attention block with an optional dropout layer and is fully
differentiable.  The ``run`` method accepts the same arguments as the
quantum counterpart (rotation_params, entangle_params, inputs) but
ignores the first two arguments, keeping the API compatible while
providing a classical baseline.

Usage
-----
>>> sa = SelfAttention(embed_dim=64, n_heads=8, dropout=0.1)
>>> out = sa.run(rotation_params=None, entangle_params=None, inputs=x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Multi‑head self‑attention block with dropout."""

    def __init__(self, embed_dim: int, n_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        if embed_dim % n_heads!= 0:
            raise ValueError("embed_dim must be divisible by n_heads")
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self,
                inputs: torch.Tensor,
                rotation_params: torch.Tensor | None = None,
                entangle_params: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute self‑attention.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, seq_len, embed_dim).
        rotation_params, entangle_params : torch.Tensor, optional
            Arguments kept for API compatibility; they are ignored by the
            classical implementation.

        Returns
        -------
        torch.Tensor
            The attended output of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = inputs.size()

        # Project to queries, keys, values
        q = self.q_proj(inputs).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(inputs).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(inputs).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention calculation
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.out_proj(out)

    def run(self, rotation_params, entangle_params, inputs):
        """Compatibility wrapper around ``forward``."""
        return self.forward(inputs, rotation_params, entangle_params)
