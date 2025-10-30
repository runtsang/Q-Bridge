"""Hybrid classical self‑attention model inspired by SelfAttention and QuantumNAT.

This implementation combines a convolutional feature extractor, a classical
self‑attention block, and a fully connected projection.  The attention
weights are computed using a lightweight “quantum‑inspired” random layer
to mimic the stochastic behaviour of a quantum circuit while staying fully
classical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHybrid(nn.Module):
    """
    Classical self‑attention module with convolutional feature extraction
    and a quantum‑inspired random projection for the attention weights.
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 64, n_heads: int = 4):
        super().__init__()
        # Feature extractor (inspired by QuantumNAT)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        # Linear projection to query/key/value
        self.qkv_proj = nn.Linear(16 * 7 * 7, 3 * embed_dim)
        # Random layer to emulate quantum parameterised rotations
        self.random_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.normal_(self.random_proj.weight, std=0.02)
        nn.init.constant_(self.random_proj.bias, 0)
        # Multi‑head attention
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        # Final projection
        self.out_proj = nn.Linear(embed_dim, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        f = self.features(x)
        f_flat = self.flatten(f)
        # Project to QKV
        qkv = self.qkv_proj(f_flat)  # shape: (bsz, 3*embed_dim)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        # Apply quantum‑inspired random projection to queries
        q = self.random_proj(q)
        # Reshape for multi‑head attention
        bsz = x.size(0)
        q = q.view(bsz, self.n_heads, self.head_dim)
        k = k.view(bsz, self.n_heads, self.head_dim)
        v = v.view(bsz, self.n_heads, self.head_dim)
        # Scaled dot‑product
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.contiguous().view(bsz, -1)
        # Final projection
        out = self.out_proj(out)
        return self.norm(out)


def SelfAttention() -> SelfAttentionHybrid:
    """Factory returning a ready‑to‑use instance."""
    return SelfAttentionHybrid()


__all__ = ["SelfAttentionHybrid", "SelfAttention"]
