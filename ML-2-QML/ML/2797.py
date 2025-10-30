"""Hybrid classical self‑attention with quanvolutional filtering."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution filter that downsamples a 28×28 image to 14×14 patches."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class HybridSelfAttentionQuanvolution(nn.Module):
    """Hybrid module combining quanvolution filtering with multi‑head self‑attention."""

    def __init__(self, embed_dim: int = 4, num_heads: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x shape: (batch, 1, 28, 28)
        features = self.filter(x)  # (batch, 4*14*14)
        seq = features.unsqueeze(1)  # (batch, 1, feature_dim)
        attn_output, _ = self.attn(seq, seq, seq)
        logits = self.linear(attn_output.squeeze(1))
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridSelfAttentionQuanvolution"]
