"""QuantumNATEnhanced: Classical backbone with multi‑head attention and a learnable embedding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Simple self‑attention that treats each feature dimension as a token.
    The output has the same dimensionality as the input.
    """
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, D) where D == embed_dim.

        Returns
        -------
        torch.Tensor
            Shape (B, D) – the attended representation.
        """
        B, D = x.shape
        seq = D  # each feature dimension becomes a token
        # Expand to (B, seq, D) so that each token has the full feature vector
        x_seq = x.unsqueeze(1).repeat(1, seq, 1)  # (B, seq, D)

        q = self.q(x_seq).reshape(B, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x_seq).reshape(B, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x_seq).reshape(B, seq, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, v)  # (B, heads, seq, head_dim)
        out = out.transpose(1, 2).reshape(B, seq, D)  # (B, seq, D)
        out = out.mean(dim=1)  # aggregate tokens
        return self.out(out)


class QuantumNATEnhanced(nn.Module):
    """
    Classical CNN backbone → multi‑head attention → projection → embedding → BatchNorm.
    This module is fully differentiable and can be trained with standard optimizers.
    """
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.attn = MultiHeadSelfAttention(embed_dim=16 * 7 * 7, num_heads=4)
        self.proj = nn.Linear(16 * 7 * 7, 64)
        self.embedding = nn.Linear(64, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, H, W) – a single‑channel image.

        Returns
        -------
        torch.Tensor
            Normalised 4‑dimensional feature vector.
        """
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        att = self.attn(flat)
        proj = self.proj(att)
        emb = self.embedding(proj)
        return self.norm(emb)


__all__ = ["QuantumNATEnhanced"]
