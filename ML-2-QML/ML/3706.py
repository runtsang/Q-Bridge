"""AdvancedEstimatorQNN – classical regression head with a transformer-based feature extractor.

This module merges a lightweight transformer encoder (borrowed and adapted from
the QTransformerTorch reference) with a simple feed‑forward head.  The
original EstimatorQNN regression network is extended by a 1‑block
transformer that operates on the two input features treated as a
sequence of length two.  The transformer extracts a compact latent
vector that is fed into a linear head for the final prediction.

Author: gpt-oss-20b
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MultiHeadAttention(nn.Module):
    """Classical multi‑head attention used inside the transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_linear(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


class _FeedForward(nn.Module):
    """Feed‑forward network used inside the transformer block."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class _TransformerBlock(nn.Module):
    """Single transformer block (attention + feed‑forward)."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = _MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = _FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class _PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class AdvancedEstimatorQNN(nn.Module):
    """Hybrid regression model that uses a transformer encoder followed
    by a linear head.  The model accepts a 2‑D input tensor and
    returns a single‑dimensional prediction."""
    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 8,
        num_heads: int = 2,
        ffn_dim: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_linear = nn.Linear(1, embed_dim)
        self.pos_encoder = _PositionalEncoder(embed_dim)
        self.transformer = _TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 2)
        x = x.unsqueeze(-1)  # (batch, 2, 1)
        x = self.input_linear(x)  # (batch, 2, embed_dim)
        x = self.pos_encoder(x)  # add positional encoding
        x = self.transformer(x)  # transformer block
        x = x.mean(dim=1)  # aggregate over sequence
        x = self.dropout(x)
        return self.head(x)


__all__ = ["AdvancedEstimatorQNN"]
