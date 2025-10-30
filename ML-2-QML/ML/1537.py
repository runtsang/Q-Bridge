"""Enhanced classical encoder with optional self‑attention and feed‑forward blocks for Quantum‑NAT."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBlock(nn.Module):
    """Simple multi‑head attention with residual connection."""
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, embed_dim]
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class FeedForwardBlock(nn.Module):
    """Position‑wise feed‑forward network with residual."""
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ff(x))


class QFCModel(nn.Module):
    """
    An advanced CNN‑Transformer hybrid that optionally adds a self‑attention block
    after the convolutional encoder. The output shape matches the original seed
    (4 features per sample).
    """
    def __init__(self, *, attention: bool = True, num_heads: int = 4) -> None:
        """
        Parameters
        ----------
        attention: bool
            Whether to insert the self‑attention block.
        num_heads: int
            Number of heads for the multi‑head attention. Ignored if attention=False.
        """
        super().__init__()
        self.attention = attention
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        if attention:
            self.attn_block = SelfAttentionBlock(embed_dim=16, num_heads=num_heads)
            self.ff_block = FeedForwardBlock(embed_dim=16, hidden_dim=64)
        else:
            self.attn_block = None
            self.ff_block = None
        self.fc = nn.Linear(16, 64)
        self.out = nn.Linear(64, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input image tensor of shape [batch, 1, 28, 28].

        Returns
        -------
        torch.Tensor
            Normalized 4‑dimensional feature vector per sample.
        """
        batch_size = x.shape[0]
        features = self.features(x)                # [B, 16, 7, 7]
        seq = features.view(batch_size, 16, -1).permute(0, 2, 1)  # [B, 49, 16]
        if self.attention:
            seq = self.attn_block(seq)
            seq = self.ff_block(seq)
        # Global average pooling over the sequence dimension
        pooled = seq.mean(dim=1)                   # [B, 16]
        hidden = F.relu(self.fc(pooled))           # [B, 64]
        out = self.out(hidden)                     # [B, 4]
        return self.norm(out)


__all__ = ["QFCModel"]
