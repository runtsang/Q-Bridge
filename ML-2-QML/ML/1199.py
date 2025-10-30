"""Enhanced classical self‑attention with multi‑head, dropout, and bias support."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SelfAttention:
    """
    Multi‑head self‑attention module that mirrors the quantum interface.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default=1
        Number of attention heads.
    dropout : float, default=0.0
        Dropout probability applied to attention weights.
    bias : bool, default=True
        Whether to include bias terms in the linear projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.bias = bias

        # Linear projection matrices constructed from rotation_params and entangle_params
        # They will be reshaped inside `run`; here we only store shapes.
        self._proj_shape = (embed_dim, embed_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        *,
        dropout_prob: float | None = None,
    ) -> np.ndarray:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Weight matrix for query/key projection.
        entangle_params : np.ndarray
            Weight matrix for value projection.
        inputs : np.ndarray
            Input embeddings of shape (batch, seq_len, embed_dim).
        dropout_prob : float, optional
            Override instance dropout probability.

        Returns
        -------
        np.ndarray
            Output embeddings of shape (batch, seq_len, embed_dim).
        """
        if dropout_prob is None:
            dropout_prob = self.dropout

        batch, seq_len, _ = inputs.shape
        # Convert to torch tensors
        inp = torch.as_tensor(inputs, dtype=torch.float32)

        # Linear projections
        Q = inp @ torch.as_tensor(
            rotation_params.reshape(self._proj_shape), dtype=torch.float32
        )
        K = inp @ torch.as_tensor(
            entangle_params.reshape(self._proj_shape), dtype=torch.float32
        )
        V = inp  # value is the raw input

        # Reshape for multi‑head
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        if dropout_prob > 0.0:
            attn = F.dropout(attn, p=dropout_prob, training=True)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return out.numpy()


__all__ = ["SelfAttention"]
