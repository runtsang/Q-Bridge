"""Hybrid quanvolution model combining classical convolution, attention, and linear head.

The module implements a classical pipeline that mirrors the quantum-inspired
architecture from the original `Quanvolution.py`.  It uses a 2×2 convolution
to extract patches, applies a multi‑head self‑attention across the patch
embeddings, and feeds the flattened representation into a linear classifier.
This design preserves the original feature dimensionality (4×14×14) while
introducing an attention mechanism that can be tuned independently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalPatchExtractor(nn.Module):
    """Extract non‑overlapping 2×2 patches from a 28×28 image."""

    def __init__(self, kernel_size: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        patches = x.unfold(2, self.kernel_size, self.kernel_size).unfold(
            3, self.kernel_size, self.kernel_size
        )  # (B, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(x.shape[0], -1, self.kernel_size**2)  # (B, 196, 4)
        return patches


class HybridQuanvolution(nn.Module):
    """Classical hybrid quanvolution with attention and linear head."""

    def __init__(self, embed_dim: int = 4, num_heads: int = 2) -> None:
        super().__init__()
        self.extractor = ClassicalPatchExtractor()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.linear = nn.Linear(embed_dim * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.extractor(x)  # (B, 196, 4)
        attn_output, _ = self.attention(patches, patches, patches)
        # flatten: (B, 196, 4) -> (B, 784)
        flat = attn_output.view(x.shape[0], -1)
        logits = self.linear(flat)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolution"]
