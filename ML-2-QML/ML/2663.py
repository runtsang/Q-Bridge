"""Hybrid self‑attention with quanvolution‑style convolution (classical)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalQuanvolutionFilter(nn.Module):
    """2×2 convolution with stride 2, mimicking the classical quanvolution filter."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the spatial dimensions after convolution
        return self.conv(x).view(x.size(0), -1)


class ClassicalSelfAttention(nn.Module):
    """Multi‑head self‑attention block using linear projections."""
    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        attn_scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ v).reshape(B, N, D)
        return self.out_proj(attn_output)


class HybridSelfAttentionQuanvolution(nn.Module):
    """Combines a quanvolution filter and a self‑attention core."""
    def __init__(self,
                 in_channels: int = 1,
                 embed_dim: int = 64,
                 num_heads: int = 4) -> None:
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter(in_channels, embed_dim // 2)
        self.attention = ClassicalSelfAttention(embed_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract spatial features
        features = self.qfilter(x)                     # (B, embed_dim//2)
        # Prepare a single‑token sequence for attention
        seq = features.unsqueeze(1)                    # (B, 1, embed_dim//2)
        # Pad to full embed_dim if necessary
        if seq.size(-1) < self.attention.embed_dim:
            pad = torch.zeros_like(seq[..., :self.attention.embed_dim - seq.size(-1)])
            seq = torch.cat([seq, pad], dim=-1)
        attn_out = self.attention(seq)                 # (B, 1, embed_dim)
        return attn_out.squeeze(1)                     # (B, embed_dim)


def SelfAttention(embed_dim: int = 64, num_heads: int = 4) -> HybridSelfAttentionQuanvolution:
    """
    Factory mirroring the original API.
    Returns an instance of the hybrid module ready for training or inference.
    """
    return HybridSelfAttentionQuanvolution(embed_dim=embed_dim, num_heads=num_heads)


__all__ = ["HybridSelfAttentionQuanvolution", "SelfAttention"]
