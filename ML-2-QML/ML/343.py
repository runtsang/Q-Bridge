"""Enhanced multi‑head self‑attention with dropout and residuals.

The class keeps the original signature but now supports multiple attention heads,
dropout, and a residual connection, providing a richer feature extractor for downstream
models.
"""

from __future__ import annotations

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
    num_heads : int
        Number of attention heads. Must divide ``embed_dim``.
    dropout : float
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs: torch.Tensor, rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mimicking the original API.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor
            Shape (embed_dim,). They are used to modulate the Q projection.
        entangle_params : torch.Tensor
            Shape (embed_dim,). They are used to modulate the K projection.

        Returns
        -------
        torch.Tensor
            Same shape as ``inputs`` – the attended representation.
        """
        batch, seq_len, _ = inputs.size()

        # Modulate Q and K with the provided parameters
        qkv = self.qkv_proj(inputs)  # (batch, seq_len, 3 * embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply parameter scaling
        q = q * rotation_params.view(1, 1, -1)
        k = k * entangle_params.view(1, 1, -1)

        # Reshape for multi‑head attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous() \
           .view(batch, seq_len, self.embed_dim)

        # Residual connection and output projection
        output = self.out_proj(attn_output) + inputs
        return output


__all__ = ["SelfAttention"]
