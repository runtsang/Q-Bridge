"""Classical multi‑head self‑attention with bias and optional dropout.

The class mirrors the original API but adds:
* configurable number of heads,
* learnable bias vectors per head,
* dropout on the attention scores,
* an optional `head_dim` argument for flexible embedding size.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention:
    """
    Multi‑head self‑attention module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    head_dim : int, default 16
        Size of each head.  Must divide embed_dim.
    dropout : float, default 0.0
        Dropout probability applied to attention logits.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        head_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        assert embed_dim % (num_heads * head_dim) == 0, (
            "embed_dim must be divisible by num_heads * head_dim"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim

        # Projection matrices for query, key, value
        self.w_q = nn.Parameter(torch.randn(embed_dim, self.total_dim))
        self.w_k = nn.Parameter(torch.randn(embed_dim, self.total_dim))
        self.w_v = nn.Parameter(torch.randn(embed_dim, self.total_dim))

        # Biases per head
        self.bias_q = nn.Parameter(torch.randn(num_heads, head_dim))
        self.bias_k = nn.Parameter(torch.randn(num_heads, head_dim))

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, L, total_dim) -> (B, num_heads, L, head_dim)."""
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, num_heads, L, head_dim) -> (B, L, total_dim)."""
        B, H, L, D = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, L, self.total_dim)

    def forward(
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
            Unused placeholder to keep interface compatible with QML.
        entangle_params : np.ndarray
            Unused placeholder to keep interface compatible with QML.
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Projections
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        # Add learnable biases
        bias_q = self.bias_q.unsqueeze(0).unsqueeze(2)  # (1, H, 1, D)
        bias_k = self.bias_k.unsqueeze(0).unsqueeze(2)
        q = self._split_heads(q) + bias_q
        k = self._split_heads(k) + bias_k
        v = self._split_heads(v)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores = self.dropout(F.softmax(scores, dim=-1))

        # Weighted sum of values
        out = torch.matmul(scores, v)  # (B, H, L, D)
        out = self._combine_heads(out)

        return out.detach().numpy()

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Backward‑compatible wrapper around ``forward``."""
        return self.forward(rotation_params, entangle_params, inputs)

__all__ = ["SelfAttention"]
