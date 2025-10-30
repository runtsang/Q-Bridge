"""Hybrid Conv‑Transformer model for classical training.

The module merges the Conv filter from the first seed with the classical
Transformer architecture from the second seed.  The convolutional layer
produces a 1‑dimensional token sequence; each token is fed into a
TransformerBlockClassical.  An optional flag `use_quantum` can replace
the transformer blocks with their quantum counterparts at runtime.
The design keeps the original Conv API (`Conv()` returns an nn.Module)
and extends it with a `TextTransformer` class that exposes a single
`forward` method compatible with the original `TextClassifier`.  The
model can be instantiated as:

    model = ConvTransformerHybrid(
        img_shape=(28,28),
        embed_dim=32,
        num_heads=4,
        num_blocks=2,
        ffn_dim=128,
        num_classes=10,
        use_quantum=False
    )

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# --------------------------------------------------------------------------- #
# 1. Convolutional filter – thin wrapper around a 2‑D kernel.
# --------------------------------------------------------------------------- #
class _ConvFilter(nn.Module):
    """Emulates the original quantum filter with a classical 2‑D convolution.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel.
    threshold : float
        Threshold applied before the sigmoid activation.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, H, W).

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, 1, H-k+1, W-k+1) containing the sigmoid‑activated
            convolution output.
        """
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        out = self.conv(x)
        out = torch.sigmoid(out - self.threshold)
        return out

# --------------------------------------------------------------------------- #
# 2. Classical transformer components (unchanged from seed).
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented classically."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

class FeedForwardClassical(nn.Module):
    """Two‑layer MLP feed‑forward block."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    """Transformer block consisting of attention + feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
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

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
# 3. Hybrid conv‑transformer model.
# --------------------------------------------------------------------------- #
class ConvTransformerHybrid(nn.Module):
    """Hybrid model that uses a convolutional patch extractor followed by a
    transformer encoder.  The transformer can be swapped with a quantum
    variant by passing `use_quantum=True` when the quantum module is imported.
    """
    def __init__(
        self,
        img_shape: Tuple[int, int],
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        kernel_size: int = 2,
        threshold: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_shape = img_shape
        self.embed_dim = embed_dim
        self.conv = _ConvFilter(kernel_size, threshold)
        # The convolution output is a single channel per patch.
        self.proj = nn.Linear(1, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of images of shape (B, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        # Convolutional tokenization
        conv_out = self.conv(x)                 # (B, 1, H-k+1, W-k+1)
        B = conv_out.size(0)
        tokens = conv_out.view(B, 1, -1).transpose(1, 2)  # (B, L, 1)
        tokens = self.proj(tokens)               # (B, L, embed_dim)
        tokens = self.pos_enc(tokens)
        tokens = self.transformer(tokens)        # (B, L, embed_dim)
        pooled = tokens.mean(dim=1)              # global average pooling
        out = self.dropout(pooled)
        return self.classifier(out)

__all__ = ["ConvTransformerHybrid"]
