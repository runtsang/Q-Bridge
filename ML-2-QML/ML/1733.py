"""Enhanced classical self‑attention with dropout and layer‑norm.

The class mirrors the original signature but introduces trainable linear layers,
dropout, and optional layer‑norm for robust experimentation.  The ``run`` method
expects the same three NumPy arrays as the seed but returns a torch tensor
ready for downstream neural‑network training.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Classical self‑attention module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input features.
    dropout : float, default 0.1
        Dropout probability applied to the attention weights.
    use_layernorm : bool, default True
        Whether to apply LayerNorm to the output.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1, use_layernorm: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_layernorm = use_layernorm

        # Trainable linear projections for query, key, value
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        if self.use_layernorm:
            self.ln = nn.LayerNorm(embed_dim)

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute self‑attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the query projection matrix (shape: embed_dim * embed_dim).
        entangle_params : np.ndarray
            Parameters for the key projection matrix (shape: embed_dim * embed_dim).
        inputs : np.ndarray
            Input tensor of shape (batch, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention‑weighted representations of shape (batch, embed_dim).
        """
        # Load parameters into linear layers
        self.query_proj.weight.data = torch.as_tensor(
            rotation_params.reshape(self.embed_dim, self.embed_dim), dtype=torch.float32
        ).t()
        self.key_proj.weight.data = torch.as_tensor(
            entangle_params.reshape(self.embed_dim, self.embed_dim), dtype=torch.float32
        ).t()

        # Forward pass
        q = self.query_proj(torch.as_tensor(inputs, dtype=torch.float32))
        k = self.key_proj(torch.as_tensor(inputs, dtype=torch.float32))
        v = self.value_proj(torch.as_tensor(inputs, dtype=torch.float32))

        scores = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        scores = self.dropout(scores)
        out = scores @ v

        if self.use_layernorm:
            out = self.ln(out)

        return out

__all__ = ["SelfAttention"]
