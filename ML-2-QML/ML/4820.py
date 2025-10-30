"""Hybrid Quanvolution classifier combining classical convolution, self‑attention, and RBF kernel."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalSelfAttention(nn.Module):
    """Dot‑product attention applied to flattened convolutional patches."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x
        k = x
        v = x
        scores = F.softmax((q @ k.T) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class RBFFunction(nn.Module):
    """Radial‑basis kernel used as an additional feature extractor."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * (diff * diff).sum(-1, keepdim=True))


class QuanvolutionHybrid(nn.Module):
    """Classical hybrid network that mirrors the quantum architecture."""
    def __init__(self, num_classes: int = 10, gamma: float = 1.0) -> None:
        super().__init__()
        # Convolutional front‑end (mirrors the original quanvolution filter)
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Self‑attention
        self.attn = ClassicalSelfAttention(embed_dim=4)
        # RBF kernel feature
        self.kernel = RBFFunction(gamma)
        # Linear head
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Convolution + patch flattening
        conv_out = self.conv(x)                         # (B, 4, 14, 14)
        patches = conv_out.view(x.size(0), -1)          # (B, 4*14*14)

        # 2. Self‑attention over patches
        attn_out = self.attn(patches)                   # (B, 4*14*14)

        # 3. Kernel similarity (here self‑kernel for illustration)
        kernel_out = self.kernel(attn_out, attn_out)    # (B, 1)

        # 4. Concatenate features
        combined = torch.cat([attn_out, kernel_out], dim=-1)

        # 5. Linear classification
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
