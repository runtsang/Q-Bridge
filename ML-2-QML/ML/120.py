"""Multi‑head self‑attention with dropout and layer normalisation, mirroring the quantum interface."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """A multi‑head self‑attention module compatible with the original interface."""
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
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
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Layer normalisation
        self.norm = nn.LayerNorm(embed_dim)

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
            Shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Attention‑weighted representations, shape (batch, seq_len, embed_dim).
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)
        # Linear projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi‑head
        batch, seq_len, _ = Q.shape
        Q = Q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        context = torch.matmul(scores, V)  # (batch, heads, seq_len, head_dim)
        context = context.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)

        # Output projection
        out = self.out_proj(context)

        # Residual + LayerNorm
        out = self.norm(out + x)
        return out.detach().numpy()


__all__ = ["SelfAttention"]
