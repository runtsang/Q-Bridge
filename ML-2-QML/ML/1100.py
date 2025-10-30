"""Enhanced classical multi-head self‑attention with optional dropout and layer normalization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionEnhanced:
    """Multi‑head self‑attention module with dropout and layer normalization.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability applied to the attention weights.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split last dimension into (num_heads, head_dim)."""
        batch, seq_len, embed_dim = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge heads back to original embedding dimension."""
        batch, num_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)

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
            Flattened weights for the linear projections (q, k, v). Shape
            (3 * embed_dim, embed_dim).
        entangle_params : np.ndarray
            Unused in the classical version but retained for API compatibility.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output of the attention layer.
        """
        # Convert to torch tensors
        x = torch.as_tensor(inputs, dtype=torch.float32)
        rot = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)

        # Apply linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi‑head attention
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        attn_output = torch.matmul(scores, v)
        attn_output = self._merge_heads(attn_output)

        # Output projection and residual connection
        output = self.out_proj(attn_output)
        output = output + x  # residual
        output = self.layer_norm(output)

        return output.numpy()

__all__ = ["SelfAttentionEnhanced"]
