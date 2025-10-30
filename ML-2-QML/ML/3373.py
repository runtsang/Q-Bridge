"""
HybridTextClassifier – classical implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Classical convolutional front‑end
# --------------------------------------------------------------------------- #
class ConvFilter1D(nn.Module):
    """1‑D convolutional feature extractor for token embeddings.

    The filter operates along the sequence dimension and is followed by a
    sigmoid activation with an optional threshold.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # (batch, out_channels, new_len) -> (batch, new_len, out_channels)
        return activations.transpose(1, 2)


# --------------------------------------------------------------------------- #
#  Transformer sub‑modules
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented in PyTorch."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return self.dropout(attn_output)


class FeedForwardClassical(nn.Module):
    """Two‑layer feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
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


class TransformerBlockClassical(nn.Module):
    """Single transformer encoder block using classical layers."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


# --------------------------------------------------------------------------- #
#  Hybrid classifier
# --------------------------------------------------------------------------- #
class HybridTextClassifier(nn.Module):
    """Transformer‑based text classifier with a classical convolutional front‑end."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        conv_kernel: int = 3,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = ConvFilter1D(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=conv_kernel,
            threshold=conv_threshold,
        )
        self.pos_encoder = PositionalEncoder(embed_dim)
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
        # token embedding
        tokens = self.token_embedding(x)  # (batch, seq_len, embed_dim)
        # conv front‑end
        conv_out = self.conv(tokens)  # (batch, new_len, embed_dim)
        # positional encoding
        x = self.pos_encoder(conv_out)
        # transformer encoder
        x = self.transformer(x)
        # pool and classify
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "ConvFilter1D",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "PositionalEncoder",
    "TransformerBlockClassical",
    "HybridTextClassifier",
]
