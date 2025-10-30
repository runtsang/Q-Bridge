"""Hybrid transformer with optional classical convolutional feature extractor.

This module keeps the classical logic from the original QTransformerTorch
while exposing a drop‑in ConvFilter that mimics the behaviour of the
quantum quanvolution layer.  The class can be instantiated with or
without the convolution stage, making it fully compatible with the
original API and easily extendable to quantum variants in the
corresponding QML module.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFilter(nn.Module):
    """
    Classical 2‑D convolution filter that emulates the quantum quanvolution
    interface.  It accepts a tensor of shape ``(B, seq_len, embed_dim)``
    and returns a tensor of the same shape after applying a single‑channel
    convolution followed by a sigmoid activation.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape ``(B, seq_len, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Convolved embeddings of shape ``(B, seq_len, embed_dim)``.
        """
        # Reshape to 4‑D: (B, C=1, H=seq_len, W=embed_dim)
        x = x.unsqueeze(1)
        out = self.conv(x)
        out = torch.sigmoid(out - self.threshold)
        # Collapse spatial dimensions back to (B, seq_len, embed_dim)
        out = out.mean(dim=[2, 3], keepdim=True)
        return out


class MultiHeadAttentionClassical(nn.Module):
    """
    Standard multi‑head attention implemented purely with PyTorch.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardClassical(nn.Module):
    """
    Two‑layer feed‑forward network used inside the transformer block.
    """

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    """
    Classical transformer block consisting of multi‑head attention and
    feed‑forward sub‑layers with residual connections and layer
    normalisation.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding used by the transformer.
    """

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class HybridTransformer(nn.Module):
    """
    Classical transformer classifier that optionally prefixes a
    convolutional feature extractor.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_dim : int
        Dimensionality of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension in the feed‑forward network.
    num_classes : int
        Number of target classes.
    dropout : float, optional
        Drop‑out probability.
    use_conv : bool, default=True
        Whether to prepend a ConvFilter.
    conv_kernel : int, default=2
        Kernel size for the convolution filter.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_conv: bool = True,
        conv_kernel: int = 2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.use_conv = use_conv
        if use_conv:
            self.conv = ConvFilter(kernel_size=conv_kernel)
            self.conv_proj = nn.Linear(1, embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input token indices of shape ``(B, seq_len)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)`` (or ``(B, 1)`` for binary).
        """
        tokens = self.token_embedding(x)          # (B, seq_len, embed_dim)
        x = self.pos_embedding(tokens)            # add positional encoding
        if self.use_conv:
            x = self.conv(x)                      # (B, seq_len, 1)
            x = self.conv_proj(x)                 # (B, seq_len, embed_dim)
        x = self.transformers(x)
        x = x.mean(dim=1)                         # global average pooling
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "HybridTransformer",
    "ConvFilter",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
]
