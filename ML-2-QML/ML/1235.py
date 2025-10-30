"""Enhanced classical self‑attention with multi‑head support and dropout.

The class can be used as a drop‑in replacement for the original
`SelfAttention` function.  It keeps the same `run` signature but
accepts a dictionary of rotation matrices for query, key and value.
"""

from __future__ import annotations

import numpy as np
import torch

class SelfAttentionPlus:
    """
    Classical multi‑head self‑attention module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 1
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability applied to the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.softmax = torch.nn.Softmax(dim=-1)

    def run(self, params: dict, inputs: np.ndarray) -> np.ndarray:
        """
        Compute multi‑head self‑attention.

        Parameters
        ----------
        params : dict
            Dictionary with keys ``q_rot``, ``k_rot`` and ``v_rot``.
            Each value is a 2‑D array of shape (embed_dim, embed_dim)
            representing the linear transformation for that head.
        inputs : np.ndarray
            Input tensor of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Output tensor of the same shape as ``inputs``.
        """
        q_rot = torch.as_tensor(params["q_rot"], dtype=torch.float32)
        k_rot = torch.as_tensor(params["k_rot"], dtype=torch.float32)
        v_rot = torch.as_tensor(params["v_rot"], dtype=torch.float32)

        # Linear projections
        Q = torch.as_tensor(inputs, dtype=torch.float32) @ q_rot
        K = torch.as_tensor(inputs, dtype=torch.float32) @ k_rot
        V = torch.as_tensor(inputs, dtype=torch.float32) @ v_rot

        # Reshape for multi‑head
        Q = Q.view(-1, self.num_heads, self.head_dim)
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = self.softmax(scores)
        if self.dropout > 0.0:
            scores = torch.nn.functional.dropout(scores, p=self.dropout, training=True)

        # Weighted sum
        out = torch.matmul(scores, V)
        out = out.reshape(-1, self.embed_dim)
        return out.numpy()

__all__ = ["SelfAttentionPlus"]
