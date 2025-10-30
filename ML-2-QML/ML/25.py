"""Classical multi‑head self‑attention with dropout and efficient tensor ops."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SelfAttentionEnhanced:
    """
    Multi‑head self‑attention module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads.
    dropout : float, default 0.1
        Dropout applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = torch.nn.Dropout(dropout)

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape to (batch, heads, seq_len, head_dim).
        """
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled dot‑product attention.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Apply multi‑head self‑attention to `inputs`.

        Parameters
        ----------
        rotation_params : np.ndarray
            Not used in the classical version but retained for API compatibility.
        entangle_params : np.ndarray
            Not used in the classical version but retained for API compatibility.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output tensor of the same shape as `inputs`.
        """
        x = torch.as_tensor(inputs, dtype=torch.float32, requires_grad=False)

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for heads
        q = self._reshape_for_heads(q)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)

        # Attention per head
        attn = self._attention(q, k, v)

        # Concatenate heads
        attn = attn.transpose(1, 2).reshape(x.shape)

        # Final projection
        out = self.out_proj(attn)

        return out.numpy()


__all__ = ["SelfAttentionEnhanced"]
