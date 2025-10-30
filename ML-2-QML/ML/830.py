"""Enhanced classical self‑attention module.

The original seed exposed a tiny, single‑head attention
implemented with PyTorch.  This extension adds
multi‑head support, dropout, and a clean API that mirrors
the quantum counterpart.  The public ``run`` method keeps
the same signature so existing downstream pipelines
continue to work unchanged.

Example
-------
>>> sa = SelfAttention(embed_dim=16, num_heads=4, dropout=0.1)
>>> out = sa.run(rotation_params=np.random.randn(16,16),
...              entangle_params=np.random.randn(16,16),
...              inputs=np.random.randn(10,16))
>>> out.shape
(10, 16)
"""

import numpy as np
import torch
import torch.nn.functional as F


class SelfAttention:
    """Multi‑head self‑attention with optional dropout."""

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to (batch, num_heads, head_dim)."""
        batch, seq_len, embed_dim = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape back to (batch, seq_len, embed_dim)."""
        batch, num_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Apply multi‑head self‑attention.

        Parameters
        ----------
        rotation_params : np.ndarray
            Weight matrix for the query projection.
        entangle_params : np.ndarray
            Weight matrix for the key projection.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            The attended representation of shape (batch, seq_len, embed_dim).
        """
        # Convert to torch tensors
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        q_w = torch.as_tensor(rotation_params, dtype=torch.float32)
        k_w = torch.as_tensor(entangle_params, dtype=torch.float32)

        # Linear projections
        query = inp @ q_w.T
        key = inp @ k_w.T
        value = inp

        # Multi‑head split
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # Scaled dot‑product
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / np.sqrt(self.head_dim)

        # Softmax + dropout
        probs = F.softmax(scores, dim=-1)
        if self.dropout > 0.0:
            probs = F.dropout(probs, p=self.dropout, training=False)

        # Weighted sum
        out = torch.matmul(probs, value)
        out = self._merge_heads(out)

        return out.numpy()


__all__ = ["SelfAttention"]
