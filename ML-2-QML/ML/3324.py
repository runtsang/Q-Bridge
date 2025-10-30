"""Hybrid classical transformer with optional quanvolution feature extractor.

This module provides a single class ``HybridTransformerClassifier`` that can
operate in two modes:

1. **Text mode** – identical to the original QTransformerTorch implementation.
2. **Image mode** – applies a classical 2‑D convolution (the analogue of the
   quantum quanvolution) to produce token embeddings that are then processed
   by the transformer.

The implementation keeps the same public API as the original anchor so that
existing training pipelines can be switched to the hybrid variant with a
single import change.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented classically."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
    """Base class for transformer blocks."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
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


class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolution that mimics the quantum quanvolution kernel."""
    def __init__(self, out_channels: int, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride)
        self.out_channels = out_channels
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, H, W]
        features = self.conv(x)                      # [batch, out_channels, H', W']
        # reshape to [batch, seq_len, out_channels]
        return features.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.out_channels)


class HybridTransformerClassifier(nn.Module):
    """Transformer‑based classifier that supports text or image input."""
    def __init__(self,
                 vocab_size: Optional[int] = None,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_blocks: int = 4,
                 ffn_dim: int = 512,
                 num_classes: int = 10,
                 dropout: float = 0.1,
                 use_quanvolution: bool = False,
                 conv_out_channels: int = 4,
                 conv_kernel: int = 2,
                 conv_stride: int = 2,
                 img_size: int = 28):
        super().__init__()
        self.use_quanvolution = use_quanvolution
        if use_quanvolution:
            self.qfilter = QuanvolutionFilter(conv_out_channels, conv_kernel, conv_stride)
            patches = (img_size - conv_kernel) // conv_stride + 1
            seq_len = patches * patches
            self.token_embedding = nn.Linear(conv_out_channels, embed_dim)
            self.seq_len = seq_len
        else:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided when use_quanvolution is False")
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.seq_len = None
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            features = self.qfilter(x)  # [batch, seq_len, out_channels]
            tokens = self.token_embedding(features)  # [batch, seq_len, embed_dim]
        else:
            tokens = self.token_embedding(x)  # [batch, seq_len, embed_dim]
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "QuanvolutionFilter",
    "HybridTransformerClassifier",
]
