"""Hybrid classical self‑attention with convolutional patch embedding."""

from __future__ import annotations

import torch
from torch import nn

class SelfAttention(nn.Module):
    """
    Classical self‑attention module that first extracts 2×2 patches
    via a 2‑D convolution, then applies a scaled dot‑product attention
    over the resulting feature maps.  The design mirrors the
    convolutional pattern used in the Quanvolution example while
    retaining the attention mechanism from the seed SelfAttention
    implementation.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 4, num_heads: int = 1) -> None:
        super().__init__()
        # 2×2 patches, stride 2 → 14×14 feature maps for MNIST
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=2, stride=2)
        self.scale = embed_dim ** -0.5
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Tensor of the same spatial shape with attention applied.
        """
        patches = self.patch_embed(x)                      # (B, D, H/2, W/2)
        B, D, H, W = patches.shape
        seq = patches.view(B, D, -1).transpose(1, 2)       # (B, N, D)

        Q = self.query_proj(seq)
        K = self.key_proj(seq)
        V = self.value_proj(seq)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).view(B, D, H, W)
        return out

__all__ = ["SelfAttention"]
