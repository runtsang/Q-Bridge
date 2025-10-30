"""Hybrid classical attention and quanvolution module.

This module merges the classical self‑attention block from SelfAttention.py
with the classical quanvolution filter from Quanvolution.py.  The
`SelfAttentionQuanvolutionHybrid` class first extracts local patches via a
2×2 convolution, then treats the flattened patch vector as a sequence and
applies a scaled dot‑product attention mechanism.  The final representation
is passed through a linear head.

The design follows the original seeds but adds a patch‑wise attention
layer, allowing the model to capture both local and global dependencies
in a single forward pass.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQuanvolutionFilter(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, embed_dim)
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)

class SelfAttentionQuanvolutionHybrid(nn.Module):
    """
    Hybrid model that applies a classical quanvolution filter followed by
    a self‑attention block and a linear classification head.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        patch_kernel: int = 2,
        patch_stride: int = 2,
        embed_dim: int = 8
    ):
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_kernel,
            stride=patch_stride
        )
        self.num_patches = 14 * 14
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim * self.num_patches, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels, height, width)
        """
        features = self.qfilter(x)
        seq = features.view(x.size(0), self.num_patches, -1)
        attn_out = self.attention(seq)
        flat = attn_out.view(x.size(0), -1)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["SelfAttentionQuanvolutionHybrid"]
