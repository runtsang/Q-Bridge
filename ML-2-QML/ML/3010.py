"""HybridQuanvolutionTransformer – classical‑only implementation.

This module implements a classical image‑to‑sequence pipeline that
mirrors the structure of the original Quanvolution and QTransformerTorch
seeds.  The filter is a lightweight 2‑D convolution that produces a
feature vector of length 4·14·14.  A linear token embedding maps this
vector to an embedding dimension, which is processed by a stack of
standard transformer blocks.  The final linear head outputs class
logits.  The design keeps the same public API as the quantum version
so that the two can be swapped at run‑time."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Classical quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilterClassic(nn.Module):
    """2‑D convolution that splits a 28×28 single‑channel image into 2×2 patches."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Input shape: (B, 1, 28, 28)
        return self.conv(x).view(x.size(0), -1)  # (B, 4*14*14)


# --------------------------------------------------------------------------- #
# 2. Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding compatible with a single token."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# 3. Classical transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockClassic(nn.Module):
    """Standard multi‑head self‑attention + feed‑forward network."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# 4. Top‑level hybrid transformer with classical filter
# --------------------------------------------------------------------------- #
class HybridQuanvolutionTransformerClassic(nn.Module):
    """Image‑to‑sequence classifier that uses a classical quanvolution filter
    followed by a stack of classical transformer blocks."""
    def __init__(
        self,
        vocab_size: int = 0,  # unused but kept for API symmetry
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilterClassic()
        self.token_embed = nn.Linear(4 * 14 * 14, embed_dim)
        self.pos_embed = PositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockClassic(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.filter(x)  # (B, 4*14*14)
        x = self.token_embed(x).unsqueeze(1)  # (B, 1, embed_dim)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)  # global average pooling over the single token
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "QuanvolutionFilterClassic",
    "PositionalEncoding",
    "TransformerBlockClassic",
    "HybridQuanvolutionTransformerClassic",
]
