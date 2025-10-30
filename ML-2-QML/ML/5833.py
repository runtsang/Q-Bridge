"""Hybrid classical self‑attention module with multi‑head, dropout, and compatibility wrapper.

The class keeps the original ``run`` interface for compatibility:
``run(rotation_params, entangle_params, inputs)``.  The rotation and entangle
parameters are interpreted as linear projection matrices for the query,
key and value vectors.  Dropout is applied to the attention scores and a
configurable number of heads is supported.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionGen231(nn.Module):
    """Extended self‑attention with multi‑head, learnable projections and dropout.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default=1
        Number of attention heads.
    dropout : float, default=0.1
        Dropout probability applied to the attention weights.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Learnable projection matrices for query, key, value
        self.W_q = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.W_k = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (num_heads, head_dim) and transpose."""
        new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.transpose(-2, -1)  # shape (..., num_heads, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of _split_heads."""
        x = x.transpose(-2, -1)
        new_shape = x.shape[:-2] + (self.embed_dim,)
        return x.reshape(*new_shape)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Standard multi‑head self‑attention forward pass."""
        q = self._split_heads(torch.matmul(inputs, self.W_q))
        k = self._split_heads(torch.matmul(inputs, self.W_k))
        v = self._split_heads(torch.matmul(inputs, self.W_v))

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        return out

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper that maps the legacy ``rotation_params`` and
        ``entangle_params`` to the learnable projection matrices.  The
        parameters are expected to be flat arrays of length ``embed_dim**2``.
        """
        if rotation_params.size!= self.embed_dim ** 2:
            raise ValueError("rotation_params size mismatch")
        if entangle_params.size!= self.embed_dim ** 2:
            raise ValueError("entangle_params size mismatch")

        # Overwrite the projection matrices with provided parameters
        self.W_q.data = torch.from_numpy(rotation_params.reshape(self.embed_dim, self.embed_dim))
        self.W_k.data = torch.from_numpy(entangle_params.reshape(self.embed_dim, self.embed_dim))
        self.W_v.data = torch.from_numpy(entangle_params.reshape(self.embed_dim, self.embed_dim))

        inputs_t = torch.from_numpy(inputs).float()
        out = self.forward(inputs_t)
        return out.detach().numpy()

__all__ = ["SelfAttentionGen231"]
