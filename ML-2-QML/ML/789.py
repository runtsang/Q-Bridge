"""Enhanced classical self‑attention with multi‑head support and dropout.

The class mirrors the quantum interface while adding richer
functionality: configurable number of heads, dropout, and
optional output of attention matrices.  It is fully
numpy‑/pytorch‑based and can be used in any classical
pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SelfAttentionBlock:
    """
    Classical multi‑head self‑attention block.

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
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        return_weights: bool = False,
    ) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for linear projections of queries.
        entangle_params : np.ndarray
            Parameters for linear projections of keys.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).
        return_weights : bool, default False
            If True, return the attention weights alongside the output.

        Returns
        -------
        output : np.ndarray
            Tensor of shape (batch, seq_len, embed_dim).
        weights : np.ndarray, optional
            Attention weights of shape (batch, num_heads, seq_len, seq_len).
        """
        batch, seq_len, _ = inputs.shape

        # Linear projections
        Q = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        K = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        V = torch.as_tensor(inputs, dtype=torch.float32)

        # Reshape for multi‑head
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        if self.dropout > 0.0:
            weights = F.dropout(weights, p=self.dropout, training=True)

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        if return_weights:
            return out.numpy(), weights.numpy()
        return out.numpy()
