"""Hybrid text classifier with optional quantum blocks.

This module implements a transformer‑based classifier that can be instantiated
in three modes:

* ``use_quantum=False`` – pure classical transformer.
* ``use_quantum=True`` – falls back to classical implementation
  (placeholder for future hybrid logic).
* ``use_conv=True`` – prepends a 2‑D convolution filter to the
  input before tokenisation, enabling simple image‑to‑text
  experiments.

Parameters
----------
vocab_size : int
    Size of the token vocabulary.
embed_dim : int
    Dimensionality of token embeddings.
num_heads : int
    Number of attention heads.
num_blocks : int
    Number of transformer blocks.
ffn_dim : int
    Dimensionality of the feed‑forward network.
num_classes : int
    Number of output classes.
dropout : float, optional
    Dropout probability.
use_quantum : bool, optional
    Flag to enable quantum sub‑modules (currently no‑op).
use_conv : bool, optional
    Flag to prepend the convolution filter.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Optional classical convolution filter (drop‑in for quanvolution)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Simple 2‑D convolution filter with a sigmoid threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> float:
        """Return mean sigmoid activation over the kernel."""
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# --------------------------------------------------------------------------- #
# Core transformer components
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
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
# Hybrid Text Classifier
# --------------------------------------------------------------------------- #
class HybridTextClassifier(nn.Module):
    """
    Transformer‑based text classifier that can operate in three modes:

    * ``use_quantum=False`` – pure classical transformer.
    * ``use_quantum=True`` – falls back to classical implementation
      (placeholder for future hybrid logic).
    * ``use_conv=True`` – prepends a 2‑D convolution filter to the
      input before tokenisation, enabling simple image‑to‑text
      experiments.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Dimensionality of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Dimensionality of the feed‑forward network.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Dropout probability.
    use_quantum : bool, optional
        Flag to enable quantum sub‑modules (currently no‑op).
    use_conv : bool, optional
        Flag to prepend the convolution filter.
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
        use_quantum: bool = False,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = ConvFilter(kernel_size=2, threshold=0.0)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        blocks = [
            TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional convolution preprocessing
        if self.use_conv:
            # Assume x is a flattened 2‑D image of shape (batch, H*W)
            # Reshape to (batch, 1, H, W) for ConvFilter
            batch, hw = x.shape
            h = w = int(hw**0.5)
            img = x.view(batch, 1, h, w)
            conv_feat = self.conv(img)
            # Use conv_feat as a single token embedding
            tokens = torch.tensor([conv_feat] * x.size(0), device=x.device)
            tokens = self.token_embedding(tokens.long())
        else:
            tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = ["HybridTextClassifier", "ConvFilter"]
