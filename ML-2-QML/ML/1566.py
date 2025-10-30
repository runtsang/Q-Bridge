"""Enhanced classical self‑attention with multi‑head support and dropout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionGen128(nn.Module):
    """
    A multi‑head self‑attention module that accepts rotation and entanglement
    parameters as weight matrices.  The interface mirrors the original
    implementation while adding dropout for regularisation and an explicit
    number of heads.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.
        num_heads : int, optional
            Number of attention heads. Defaults to 4.
        dropout : float, optional
            Dropout probability applied to the attention output. Defaults to 0.1.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear layers for query, key, value will be constructed from
        # rotation_params and entangle_params during the run call.
        # We keep placeholders for shape consistency.
        self.register_buffer("q_weight_placeholder", torch.empty(0))
        self.register_buffer("k_weight_placeholder", torch.empty(0))
        self.register_buffer("v_weight_placeholder", torch.empty(0))

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
            Shape (embed_dim, embed_dim).  Interpreted as the weight matrix
            for the query projection.
        entangle_params : np.ndarray
            Shape (embed_dim, embed_dim).  Used as the weight matrix for
            the key projection.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            The attended representation, shape (batch, seq_len, embed_dim).
        """
        # Convert to torch tensors
        x = torch.as_tensor(inputs, dtype=torch.float32)
        q_w = torch.as_tensor(rotation_params, dtype=torch.float32)
        k_w = torch.as_tensor(entangle_params, dtype=torch.float32)
        v_w = torch.eye(self.embed_dim, dtype=torch.float32)

        # Linear projections
        Q = torch.einsum("bse,ef->bshf", x, q_w).reshape(
            x.shape[0], x.shape[1], self.num_heads, self.head_dim
        )
        K = torch.einsum("bse,ef->bshf", x, k_w).reshape(
            x.shape[0], x.shape[1], self.num_heads, self.head_dim
        )
        V = torch.einsum("bse,ef->bshf", x, v_w).reshape(
            x.shape[0], x.shape[1], self.num_heads, self.head_dim
        )

        # Scaled dot‑product attention
        scores = torch.einsum("bshi,bshj->bhsij", Q, K) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.einsum("bhsij,bshj->bshi", attn, V)
        out = out.reshape(x.shape[0], x.shape[1], self.embed_dim)
        return out.numpy()


__all__ = ["SelfAttentionGen128"]
