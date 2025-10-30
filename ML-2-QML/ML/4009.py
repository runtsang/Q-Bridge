"""Hybrid classical self‑attention transformer.

This module implements a purely classical transformer block that can be
used as a drop‑in replacement for the original SelfAttention.py.
The design mirrors the QTransformerTorch implementation but keeps all
operations on the CPU.  The class name is intentionally identical to the
quantum variant so that the two modules can be swapped at import time.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SelfAttentionTransformer"]


class BaseAttention(nn.Module):
    """Shared logic for multi‑head attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.embed_dim)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        return self.dropout(F.softmax(scores, dim=-1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class ClassicalAttention(BaseAttention):
    """Standard multi‑head self‑attention implemented purely in PyTorch."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        attn = self._attn(q, k, v, mask)
        return self.out_proj(self._merge_heads(attn))


class FeedForwardBase(nn.Module):
    """Base for feed‑forward layers."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class SelfAttentionTransformer(nn.Module):
    """Hybrid transformer block that is fully classical."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 bias: bool = False) -> None:
        super().__init__()
        self.attn = ClassicalAttention(embed_dim, num_heads, dropout, bias)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))
