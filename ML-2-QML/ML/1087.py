"""Enhanced classical self‑attention with multi‑head and dropout."""

import numpy as np
import torch
import torch.nn.functional as F

class ClassicalSelfAttention:
    """Multi‑head self‑attention with optional dropout and residual."""
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.dropout = torch.nn.Dropout(dropout)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (embed_dim, 3) – linear weights for queries, keys and values.
        entangle_params : np.ndarray
            Shape (embed_dim,) – bias terms applied to keys.
        inputs : np.ndarray
            Shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Shape (batch, seq_len, embed_dim) – attended representations.
        """
        batch, seq_len, _ = inputs.shape

        # Linear projections
        queries = torch.from_numpy(inputs @ rotation_params[:, 0:1]).float()
        keys    = torch.from_numpy(inputs @ rotation_params[:, 1:2] + entangle_params).float()
        values  = torch.from_numpy(inputs).float()

        # Reshape for multi‑head
        queries = queries.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        keys    = keys.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        values  = values.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # Scaled dot‑product attention
        scores = torch.matmul(queries, keys.transpose(-2,-1)) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        out = torch.matmul(scores, values)
        out = out.transpose(1,2).contiguous().view(batch, seq_len, self.embed_dim)
        return out.numpy()

def SelfAttention(embed_dim: int = 4, num_heads: int = 2, dropout: float = 0.1):
    """Factory returning a ClassicalSelfAttention instance."""
    return ClassicalSelfAttention(embed_dim, num_heads, dropout)

__all__ = ["SelfAttention"]
