"""Hybrid vision transformer with classical front‑end and optional quantum transformer blocks.

This module extends the original ``TextClassifier`` to operate on image data.  
It introduces a configurable convolutional patch extractor and a choice of classical or quantum transformer layers.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Multi‑Head Attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   batch_size: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch's MultiheadAttention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 mask: Optional[torch.Tensor] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


# --------------------------------------------------------------------------- #
#  Feed‑Forward Network
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    """Shared interface for feed‑forward layers."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# --------------------------------------------------------------------------- #
#  Transformer Block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Purely classical transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Positional Encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Classical Patch Extractor (Convolutional front‑end)
# --------------------------------------------------------------------------- #
class ClassicalPatchExtractor(nn.Module):
    """Extracts non‑overlapping 2×2 patches via a 2‑D convolution."""

    def __init__(self, in_channels: int, patch_dim: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, patch_dim * patch_dim,
                              kernel_size=patch_dim, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Shape: (B, patch_dim*patch_dim, H/2, W/2)
        patches = self.conv(x)
        # Flatten patches into sequence
        return patches.view(x.size(0), -1, patches.size(1))


# --------------------------------------------------------------------------- #
#  Hybrid Transformer Classifier
# --------------------------------------------------------------------------- #
class HybridTransformerClassifier(nn.Module):
    """Vision transformer with optional quantum front‑end and transformer layers.

    Parameters
    ----------
    input_channels : int
        Number of image channels (default 1 for grayscale).
    embed_dim : int
        Embedding dimension for transformer tokens.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of feed‑forward network.
    num_classes : int
        Number of target classes.
    dropout : float
        Drop‑out probability.
    use_quantum_front : bool
        If ``True`` the quantum quanvolution filter is used as front‑end.
    use_quantum_transformer : bool
        If ``True`` each transformer block is quantum‑enhanced.
    """

    def __init__(self,
                 input_channels: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum_front: bool = False,
                 use_quantum_transformer: bool = False) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ffn_dim = ffn_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_quantum_front = use_quantum_front
        self.use_quantum_transformer = use_quantum_transformer

        # Front‑end: classical or quantum
        if use_quantum_front:
            # Quantum front‑end is defined in the quantum module
            raise NotImplementedError("Quantum front‑end must be instantiated in the quantum module.")
        else:
            self.front = ClassicalPatchExtractor(input_channels)

        # Positional encoding
        self.pos_enc = PositionalEncoder(embed_dim)

        # Transformer blocks
        blocks = []
        for _ in range(num_blocks):
            if use_quantum_transformer:
                # Quantum transformer block is defined in the quantum module
                raise NotImplementedError("Quantum transformer block must be instantiated in the quantum module.")
            else:
                blocks.append(TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout))
        self.transformers = nn.Sequential(*blocks)

        # Classifier head
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patches
        patches = self.front(x)  # shape: (B, seq_len, patch_dim*patch_dim)
        # Project to embed_dim
        x = nn.Linear(patches.size(-1), self.embed_dim)(patches)
        # Positional encoding
        x = self.pos_enc(x)
        # Transformer
        x = self.transformers(x)
        # Pool and classify
        x = x.mean(dim=1)  # global average pooling
        x = self.dropout_layer(x)
        return self.classifier(x)


# Alias for backward compatibility with the original TextClassifier
TextClassifier = HybridTransformerClassifier

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "ClassicalPatchExtractor",
    "HybridTransformerClassifier",
    "TextClassifier",
]
