"""Classical multi-head self‑attention layer with optional dropout.
The interface mirrors the original SelfAttention() function while adding richer
capabilities such as multiple heads and dropout, useful for hybrid training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    """
    Multi‑head self‑attention module compatible with the legacy
    ``SelfAttention()`` interface.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, optional
        Number of attention heads.  Defaults to 1.
    dropout : float, optional
        Drop‑out probability applied to the attention weights.  Defaults
        to 0.0.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"
        self.dropout = nn.Dropout(dropout)

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """
        Forward pass using the legacy parameter interface.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params : np.ndarray
            Parameters reshaped to (embed_dim, -1) used for the query
            projection.
        entangle_params : np.ndarray
            Parameters reshaped to (embed_dim, -1) used for the key
            projection.
        """
        # Apply linear projections
        q = self.q_proj(inputs)  # (batch, seq, embed)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)

        # Optional rotation and entangle emulation
        q = torch.matmul(
            q, torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=q.dtype)
        )
        k = torch.matmul(
            k, torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=k.dtype)
        )

        # Reshape for multi‑head
        batch, seq, _ = q.size()
        q = q.view(batch, seq, self.num_heads, self.head_dim)
        k = k.view(batch, seq, self.num_heads, self.head_dim)
        v = v.view(batch, seq, self.num_heads, self.head_dim)

        # Scaled dot‑product attention
        scores = torch.einsum("bshd,bshd->bhs", q, k) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        context = torch.einsum("bhs,bshd->bshd", scores, v)
        context = context.reshape(batch, seq, self.embed_dim)

        return self.out_proj(context)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Legacy ``run`` method that accepts NumPy inputs and returns a NumPy
        array, allowing seamless replacement of the original implementation.
        """
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        out_t = self.forward(inputs_t, rotation_params, entangle_params)
        return out_t.detach().numpy()


def SelfAttention() -> SelfAttentionLayer:
    """
    Factory function maintaining compatibility with the original
    ``SelfAttention`` module.  Returns a layer with default parameters.
    """
    return SelfAttentionLayer(embed_dim=4, num_heads=1, dropout=0.0)


__all__ = ["SelfAttention", "SelfAttentionLayer"]
