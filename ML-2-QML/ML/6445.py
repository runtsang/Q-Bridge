"""Hybrid classical transformer with convolutional backbone.

This module implements a text‑style transformer that first extracts visual
features using a small CNN and then processes them with a stack of
classical transformer blocks.  It shares the public API of the original
`TextClassifier` so it can be dropped into existing pipelines.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFeatureExtractor(nn.Module):
    """Simple CNN that produces a sequence of token embeddings."""

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # -> (8, H, W)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (8, H/2, W/2)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # -> (16, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (16, H/4, W/4)
        )
        self.embed_dim = embed_dim
        # Linear projection from 16 channels to embed_dim
        self.proj = nn.Linear(16, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (B, 1, H, W)
        Returns:
            tokens: (B, seq_len, embed_dim) where seq_len = (H/4)*(W/4)
        """
        batch, _, h, w = x.shape
        feats = self.features(x)  # (B, 16, h/4, w/4)
        seq_len = feats.shape[2] * feats.shape[3]
        feats = feats.view(batch, 16, -1).transpose(1, 2)  # (B, seq_len, 16)
        tokens = self.proj(feats)  # (B, seq_len, embed_dim)
        return tokens


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardClassical(nn.Module):
    """Two‑layer MLP."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    """Single transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridTransformerCNNClassifier(nn.Module):
    """Hybrid CNN + transformer classifier."""

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 16,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cnn = ConvFeatureExtractor(embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(
                    embed_dim, num_heads, ffn_dim, dropout
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor of shape (B, 1, H, W)
        Returns:
            logits: (B, num_classes) or (B, 1) for binary
        """
        tokens = self.cnn(x)           # (B, seq_len, embed_dim)
        tokens = self.pos_enc(tokens)
        x = self.transformer(tokens)
        x = x.mean(dim=1)              # global pooling
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "ConvFeatureExtractor",
    "PositionalEncoder",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "HybridTransformerCNNClassifier",
]
