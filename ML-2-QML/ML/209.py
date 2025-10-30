"""Enhanced classical self‑attention with multi‑head capability and trainable layers.

The class mirrors the original interface but now supports multiple heads,
dropout, and a small neural head to learn the query/key/value projections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalSelfAttention(nn.Module):
    """Multi‑head self‑attention module compatible with the original API.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, optional
        Number of attention heads. Defaults to 4.
    dropout : float, optional
        Dropout probability applied to the attention weights. Defaults to 0.1.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Learnable projection matrices for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Weight matrix for query projection reshaped to (embed_dim, embed_dim).
        entangle_params : np.ndarray
            Weight matrix for key projection reshaped to (embed_dim, embed_dim).
        inputs : np.ndarray
            Input embeddings of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output embeddings after multi‑head attention.
        """
        # Convert to torch tensors
        X = torch.as_tensor(inputs, dtype=torch.float32)
        Q = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        K = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)

        # Apply learnable projections
        Q = self.q_proj(X) @ Q
        K = self.k_proj(X) @ K
        V = self.v_proj(X)

        # Reshape for multi‑head
        batch, seq_len, _ = Q.shape
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return out.detach().numpy()


def SelfAttention() -> ClassicalSelfAttention:
    """Return a pre‑configured multi‑head attention instance."""
    return ClassicalSelfAttention(embed_dim=4, num_heads=4, dropout=0.0)


__all__ = ["SelfAttention"]
