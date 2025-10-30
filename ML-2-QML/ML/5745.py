"""Enhanced classical model for Quantum‑NAT experiments.

This model extends the original CNN by adding a lightweight multi‑head
self‑attention block after the second convolution.  The attention
re‑weights spatial features before the final fully‑connected projection.
The architecture remains fully classical and returns a 4‑dimensional
feature vector suitable for downstream quantum‑classical experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Simple multi‑head attention over flattened spatial features."""
    def __init__(self, embed_dim: int, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, h, w)
        batch, c, h, w = x.shape
        x_flat = x.view(batch, c, -1).transpose(1, 2)  # (batch, seq, c)
        qkv = self.qkv(x_flat)  # (batch, seq, 3*embed)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, -1, self.num_heads * self.head_dim)
        out = self.out_proj(attn_out)
        return out.transpose(1, 2).view(batch, c, h, w)

class QuantumNATEnhanced(nn.Module):
    """Hybrid classical‑only model with attention‑augmented CNN."""
    def __init__(self, num_classes: int = 4, num_heads: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.attn = MultiHeadAttention(embed_dim=16, num_heads=num_heads)
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        feat = self.attn(feat)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
