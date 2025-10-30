"""Hybrid classical convolution–transformer–regressor module.

This module fuses ideas from the original Conv, QTransformerTorch, and EstimatorQNN
implementations.  The class can be used as a drop‑in replacement for the
quantum‑aware counterparts while remaining fully classical.  It demonstrates
how a convolutional front‑end, a stack of transformer blocks, and a
fully‑connected head can be composed into a single end‑to‑end model.

The public API mirrors the quantum version: the constructor accepts the same
hyper‑parameters and the ``forward`` method produces a scalar (or
multiclass) output.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SimpleTransformerBlock(nn.Module):
    """A lightweight transformer block built from PyTorch primitives."""

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
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridConvTransformerEstimator(nn.Module):
    """
    Classical model that mimics the behaviour of the quantum‑centric
    architecture.  It contains:

    * a 2‑D convolutional filter (drop‑in replacement for quanvolution)
    * a stack of transformer blocks
    * a final fully‑connected head
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_blocks: int = 2,
        ffn_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Convolution front‑end
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Transformer stack
        self.transformer = nn.Sequential(
            *[
                SimpleTransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        # Projection from 1‑dim to embed_dim
        self.proj = nn.Linear(1, embed_dim)

        # Fully‑connected head
        self.fc = nn.Linear(embed_dim, num_classes if num_classes > 1 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, height, width).

        Returns
        -------
        torch.Tensor
            Model output.  Shape is (batch, 1) for regression or
            (batch, num_classes) for classification.
        """
        # Convolution
        x = x.unsqueeze(1)  # add channel dimension
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        pooled = activations.mean([2, 3])  # global average pooling

        # Prepare sequence for transformer: (batch, seq_len=1, embed_dim)
        seq = pooled.unsqueeze(1)  # (batch, 1, 1)
        seq = self.proj(seq)       # (batch, 1, embed_dim)

        # Transformer
        trans_out = self.transformer(seq)  # (batch, 1, embed_dim)
        trans_out = trans_out.mean(dim=1)  # (batch, embed_dim)

        # Head
        return self.fc(trans_out)


__all__ = ["HybridConvTransformerEstimator"]
