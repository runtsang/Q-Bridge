"""Enhanced multi‑head self‑attention implementation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def SelfAttention():
    class MultiHeadSelfAttention:
        """
        Multi‑head self‑attention with optional dropout.
        Parameters are interpreted as linear projections for Q, K and V.
        """

        def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.dropout = torch.nn.Dropout(dropout)

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
        ) -> np.ndarray:
            """
            Parameters
            ----------
            rotation_params : np.ndarray
                Shape (embed_dim, num_heads) – linear weights for Q.
            entangle_params : np.ndarray
                Shape (embed_dim, num_heads) – linear weights for K and V.
            inputs : np.ndarray
                Shape (batch, seq_len, embed_dim).

            Returns
            -------
            np.ndarray
                Output of the multi‑head attention, shape (batch, seq_len, embed_dim).
            """
            batch, seq_len, _ = inputs.shape
            # Convert to tensors
            inp = torch.as_tensor(inputs, dtype=torch.float32)

            # Linear projections
            Q = inp @ torch.as_tensor(rotation_params, dtype=torch.float32)
            K = inp @ torch.as_tensor(entangle_params, dtype=torch.float32)
            V = inp @ torch.as_tensor(entangle_params, dtype=torch.float32)

            # Reshape for heads
            Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled dot‑product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            scores = F.softmax(scores, dim=-1)
            scores = self.dropout(scores)

            # Weighted sum
            out = torch.matmul(scores, V)

            # Concatenate heads
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
            return out.numpy()

    return MultiHeadSelfAttention(embed_dim=4, num_heads=2, dropout=0.1)


__all__ = ["SelfAttention"]
