"""
Multi‑head self‑attention module for classical deep learning.

The class encapsulates a PyTorch‑based implementation that mirrors the
original single‑head design while adding:
  • Support for multiple attention heads.
  • Configurable dropout for regularisation.
  • A concise `run` method that accepts rotation & entangle weight matrices
    and returns the attended representation as a NumPy array.

Typical usage:

    from SelfAttention import SelfAttention
    attention = SelfAttention(embed_dim=128, num_heads=8)
    out = attention.run(inputs, q_weights, k_weights, dropout=0.1)

"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.functional import softmax

class SelfAttention:
    """Multi‑head self‑attention implementation in PyTorch."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.
        num_heads : int
            Number of attention heads.
        dropout : float, default 0.0
            Dropout probability applied to the attention scores.
        """
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Dropout layer
        self.drop = torch.nn.Dropout(p=dropout)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass of the multi‑head attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Query weight matrix of shape
            (num_heads, head_dim, embed_dim).  These are analogous to the
            rotation parameters in the seed.
        entangle_params : np.ndarray
            Key weight matrix of shape
            (num_heads, head_dim, embed_dim).  Analogous to entangle params.
        inputs : np.ndarray
            Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape

        # Convert to torch tensors
        q_w = torch.as_tensor(rotation_params, dtype=torch.float32)
        k_w = torch.as_tensor(entangle_params, dtype=torch.float32)
        inp = torch.as_tensor(inputs, dtype=torch.float32)

        # Compute queries, keys, values per head
        queries = torch.einsum("bse,hde->bhse", inp, q_w)  # (batch, heads, seq, head_dim)
        keys = torch.einsum("bse,hde->bhse", inp, k_w)
        values = inp.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # (batch, heads, seq, embed_dim)

        # Scaled dot‑product attention
        scores = torch.einsum("bhqd,bhkd->bhqk", queries, keys) / np.sqrt(self.head_dim)
        scores = softmax(scores, dim=-1)
        scores = self.drop(scores)

        # Weighted sum of values
        attended = torch.einsum("bhqk,bhkd->bhqd", scores, values)  # (batch, heads, seq, head_dim)

        # Concatenate heads
        attended = attended.reshape(batch, seq_len, self.embed_dim)

        return attended.detach().numpy()

__all__ = ["SelfAttention"]
