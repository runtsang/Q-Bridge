"""Enhanced classical self‑attention module with multi‑head support and dropout.

The class retains the original `run` signature:
    run(rotation_params, entangle_params, inputs)
where `rotation_params` and `entangle_params` are interpreted as weight matrices
for the query/key and value projections respectively.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention with optional dropout.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default=1
        Number of attention heads.
    dropout : float, default=0.0
        Dropout probability applied to the attention weights.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass compatible with the original interface.

        Parameters
        ----------
        rotation_params : np.ndarray
            Weight matrix for Q and K projections. Shape: (embed_dim, embed_dim)
        entangle_params : np.ndarray
            Weight matrix for V projection. Shape: (embed_dim, embed_dim)
        inputs : np.ndarray
            Input embeddings. Shape: (batch_size, embed_dim)

        Returns
        -------
        np.ndarray
            Output after self‑attention. Shape: (batch_size, embed_dim)
        """
        # Load weights from the provided numpy arrays
        self.q_proj.weight.data = torch.tensor(rotation_params, dtype=torch.float32)
        self.k_proj.weight.data = torch.tensor(rotation_params, dtype=torch.float32)
        self.v_proj.weight.data = torch.tensor(entangle_params, dtype=torch.float32)

        x = torch.tensor(inputs, dtype=torch.float32)

        # Linear projections
        Q = self.q_proj(x)  # (B, D)
        K = self.k_proj(x)  # (B, D)
        V = self.v_proj(x)  # (B, D)

        # Reshape for multi‑head
        B = Q.size(0)
        Q = Q.view(B, self.num_heads, self.head_dim).transpose(0, 1)  # (H, B, D_h)
        K = K.view(B, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(B, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (H, B, B)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        context = torch.matmul(scores, V)  # (H, B, D_h)
        context = context.transpose(0, 1).contiguous().view(B, -1)  # (B, D)

        # Final linear projection
        out = self.out_proj(context)  # (B, D)
        return out.detach().numpy()

    # Alias to maintain original interface
    run = forward


__all__ = ["SelfAttention"]
