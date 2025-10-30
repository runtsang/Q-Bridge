"""Hybrid classical self‑attention module combining patch extraction and multi‑head attention.

Features
--------
- Patch extraction via a 2×2 convolution (inspired by Quanvolution).
- Multi‑head self‑attention over the flattened patches.
- Optional quantum‑inspired kernel (placeholder) can be plugged in externally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbedding(nn.Module):
    """Extract 2×2 patches and embed them into a 4‑dimensional space."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, H, W)
        return self.conv(x)  # (batch, out_channels, H', W')

class MultiHeadSelfAttention(nn.Module):
    """Classic multi‑head self‑attention using PyTorch's implementation."""
    def __init__(self, embed_dim: int, num_heads: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        return attn_output

class SelfAttention(nn.Module):
    """
    Hybrid self‑attention module that first extracts patch embeddings and then
    applies multi‑head self‑attention. The output has the same shape as the
    input tensor.
    """
    def __init__(self, embed_dim: int = 4, num_heads: int = 2) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(out_channels=embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch extraction
        patches = self.patch_embed(x)  # (batch, embed_dim, H', W')
        batch, embed_dim, h, w = patches.shape
        seq_len = h * w
        # Flatten patches
        patches = patches.view(batch, embed_dim, seq_len)  # (batch, embed_dim, seq_len)
        # Prepare for MultiheadAttention: (seq_len, batch, embed_dim)
        patches = patches.permute(2, 0, 1)
        # Self‑attention
        attn_out = self.attn(patches)  # (seq_len, batch, embed_dim)
        # Project back
        attn_out = self.proj(attn_out)
        # Reshape to original spatial layout
        attn_out = attn_out.permute(1, 2, 0).contiguous()  # (batch, embed_dim, seq_len)
        attn_out = attn_out.view(batch, embed_dim, h, w)
        return attn_out

__all__ = ["SelfAttention", "PatchEmbedding", "MultiHeadSelfAttention"]
