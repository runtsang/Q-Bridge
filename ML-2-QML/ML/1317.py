"""Enhanced classical self‑attention with a learnable projection and dropout."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionEnhanced:
    """A hybrid‑style self‑attention block that can be dropped into a neural network."""

    def __init__(self, embed_dim: int, head_count: int = 1, dropout: float = 0.0, **kwargs):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.
        head_count : int, optional
            Number of attention heads. Defaults to 1.
        dropout : float, optional
            Dropout probability applied after the attention output. Defaults to 0.0.
        """
        self.embed_dim = embed_dim
        self.head_count = head_count
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        # Learnable linear projection before attention
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Project input
        proj_x = self.proj(x)
        # Compute queries, keys, values
        Q = torch.einsum('bse,ef->bsf', proj_x, torch.eye(self.embed_dim, device=x.device))
        K = torch.einsum('bse,ef->bsf', proj_x, torch.eye(self.embed_dim, device=x.device))
        V = proj_x
        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        # Apply dropout and return
        return self.dropout(attn_output)


__all__ = ["SelfAttentionEnhanced"]
