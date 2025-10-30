"""Enhanced multi‑head self‑attention using PyTorch."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    """
    Multi‑head self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads.
    dropout : float, optional
        Drop‑out probability applied to attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output linear projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute the self‑attention output.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        rotation_params : np.ndarray
            Shape (embed_dim, embed_dim) – used to initialise the query projection.
        entangle_params : np.ndarray
            Shape (embed_dim, embed_dim) – used to initialise the key projection.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        # Initialise projections from external parameters
        self.q_proj.weight.data = torch.from_numpy(rotation_params).to(
            self.q_proj.weight.dtype
        )
        self.k_proj.weight.data = torch.from_numpy(entangle_params).to(
            self.k_proj.weight.dtype
        )

        B, N, _ = inputs.shape

        # Project to queries, keys, values
        q = self.q_proj(inputs).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(inputs).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(inputs).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, N, self.embed_dim)

        # Residual connection + output projection
        out = self.out_proj(out) + inputs
        return out


__all__ = ["SelfAttentionModule"]
