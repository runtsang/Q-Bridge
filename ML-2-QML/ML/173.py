"""Enhanced multi‑head self‑attention with dropout and layernorm.

This class extends the original single‑head attention by adding:
* 4‑head default projection (configurable).
* Dropout applied to the value vectors.
* LayerNorm after the multi‑head aggregation.
* Input reshaping helper to convert a 3‑D array into a 2‑D matrix.

The interface matches the original: run(rotation_params, entangle_params, inputs)
where `rotation_params` and `entangle_params` are used to build the linear
projection matrices; `inputs` is a NumPy array of shape (batch, seq, embed).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention__gen191:
    """Multi‑head self‑attention with optional dropout and layernorm."""

    def __init__(
        self,
        embed_dim: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim) if use_layer_norm else None

    def _reshape_to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, seq, embed) -> (batch, seq, heads, head_dim)."""
        return tensor.view(tensor.shape[0], tensor.shape[1], self.num_heads, self.head_dim)

    def _reshape_from_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, seq, heads, head_dim) -> (batch, seq, embed)."""
        return tensor.view(tensor.shape[0], tensor.shape[1], self.embed_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (embed_dim * embed_dim,).  Used to build query, key, value
            projection matrices.
        entangle_params : np.ndarray
            Shape (embed_dim * embed_dim,).  Used identically to rotation_params
            for key and value projections.
        inputs : np.ndarray
            Shape (batch, seq, embed_dim).

        Returns
        -------
        np.ndarray
            Attention output with the same shape as `inputs`.
        """
        batch, seq, _ = inputs.shape

        # Build projection matrices
        W_q = torch.as_tensor(
            rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        W_k = torch.as_tensor(
            entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        # Linear projections
        queries = torch.as_tensor(inputs, dtype=torch.float32) @ W_q
        keys = torch.as_tensor(inputs, dtype=torch.float32) @ W_k
        values = torch.as_tensor(inputs, dtype=torch.float32)  # identity

        # Reshape to heads
        queries = self._reshape_to_heads(queries)
        keys = self._reshape_to_heads(keys)
        values = self._reshape_to_heads(values)

        # Scaled dot‑product attention per head
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)

        # Apply dropout to values before weighting
        values = self.dropout(values)

        # Weighted sum
        context = torch.matmul(scores, values)
        context = self._reshape_from_heads(context)

        # Optional layer norm
        if self.layer_norm is not None:
            context = self.layer_norm(context)

        return context.numpy()

__all__ = ["SelfAttention__gen191"]
