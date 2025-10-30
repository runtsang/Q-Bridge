"""Hybrid transformer with optional classical CNN feature extractor.

This module mirrors the API of the quantum variant so that the two
implementations can be swapped at run‑time.  The design follows the
structure of the original `QTransformerTorch.py` seed but adds an
optional CNN front‑end for image data and a lightweight CNN
implementation (`ClassicalCNN`) that is used when `use_cnn=True`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Low‑level transformer primitives – classical
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention layers."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        out = torch.matmul(scores, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(out)


class FeedForwardBase(nn.Module):
    """Base class for the position‑wise feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """2‑layer MLP with ReLU."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# --------------------------------------------------------------------------- #
#  Transformer block and positional encoding
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base class for a transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Optional classical CNN feature extractor
# --------------------------------------------------------------------------- #
class ClassicalCNN(nn.Module):
    """2‑layer CNN that maps 1‑channel images to a dense embedding."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Assume input 28×28 → 7×7 after two pooling layers
        self.fc = nn.Linear(32 * 7 * 7, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# --------------------------------------------------------------------------- #
#  Public API – classical hybrid transformer
# --------------------------------------------------------------------------- #
class QTransformerGen(nn.Module):
    """
    Classical transformer‑based classifier with an optional CNN feature extractor.
    The architecture is intentionally identical to the quantum variant so that
    a downstream pipeline can instantiate either implementation via the same
    constructor signature.
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        ffn_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_cnn: bool = False,
    ) -> None:
        super().__init__()
        self.use_cnn = use_cnn

        if use_cnn:
            if vocab_size is not None:
                raise ValueError("vocab_size is ignored when use_cnn=True")
            self.feature_extractor = ClassicalCNN(embed_dim)
        else:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided for tokenised input")
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            * For tokenised text: shape (B, seq_len)
            * For images: shape (B, 1, H, W)

        Returns
        -------
        logits : torch.Tensor
            Shape (B, num_classes) or (B, 1) for binary classification.
        """
        if self.use_cnn:
            x = self.feature_extractor(x)          # (B, embed_dim)
            x = x.unsqueeze(1)                    # (B, 1, embed_dim)
        else:
            x = self.token_embedding(x)           # (B, seq_len, embed_dim)

        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)                          # global average pooling over seq_len
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "QTransformerGen",
    "ClassicalCNN",
    "PositionalEncoder",
    "TransformerBlockClassical",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
]
