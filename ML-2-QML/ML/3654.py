"""Transformer‑based classical sampler that mirrors the original SamplerQNN API."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for consistency with transformer designs."""

    def __init__(self, embed_dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """Standard multi‑head attention implemented classically."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        q = self.w_q(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Simple feed‑forward block used inside transformer."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with attention, feed‑forward and residuals."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class _SamplerQNN(nn.Module):
    """Transformer‑based sampler that maps 2‑dimensional inputs to a categorical distribution over 2 classes."""

    def __init__(
        self,
        embed_dim: int = 2,
        num_heads: int = 1,
        ffn_dim: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoder(embed_dim)
        self.block = TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
        self.proj = nn.Linear(embed_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, 2)
        x = self.pos_enc(inputs.unsqueeze(1))  # add sequence dimension
        x = self.block(x)
        logits = self.proj(x.squeeze(1))
        return F.softmax(logits, dim=-1)


def SamplerQNN() -> _SamplerQNN:
    """Compatibility wrapper mirroring the original API."""
    return _SamplerQNN()


__all__ = ["SamplerQNN"]
