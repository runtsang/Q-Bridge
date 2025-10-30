"""Enhanced classical self‑attention with multi‑head support and optional feed‑forward."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalSelfAttentionGen192:
    """
    Multi‑head self‑attention module that mirrors the quantum interface.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied to attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.dropout = nn.Dropout(dropout)

        # Output projection – identity for a lightweight forward pass
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj.weight.data = torch.eye(embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim) and transpose.
        """
        batch, seq_len, embed_dim = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch, heads, seq_len, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads back to original embedding dimension.
        """
        batch, heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim)
        return x

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute self‑attention output.
        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the linear projections. Shape should be
            (embed_dim, embed_dim * 3) when flattened.
        entangle_params : np.ndarray
            Parameters for optional entanglement (unused in this classical version).
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).
        Returns
        -------
        np.ndarray
            Output of shape (batch, seq_len, embed_dim).
        """
        # Convert to torch tensors
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32, requires_grad=True)

        # Load rotation parameters into projection matrices
        # We interpret the first embed_dim*embed_dim entries as a single weight matrix
        # and reuse it for Q, K, V for simplicity.
        weight = torch.as_tensor(
            rotation_params[: self.embed_dim * self.embed_dim]
           .reshape(self.embed_dim, self.embed_dim),
            dtype=torch.float32,
        )

        # Linear projections
        Q = F.linear(inputs_t, weight)
        K = F.linear(inputs_t, weight)
        V = F.linear(inputs_t, weight)

        # Split heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        context = torch.matmul(attn_weights, V)
        context = self._merge_heads(context)

        # Final linear projection
        output = F.linear(context, self.out_proj.weight)

        return output.detach().cpu().numpy()


def SelfAttention():
    """
    Factory that returns a ClassicalSelfAttentionGen192 instance with default parameters.
    """
    return ClassicalSelfAttentionGen192(embed_dim=192)
