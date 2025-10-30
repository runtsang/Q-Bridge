"""Hybrid classical quanvolution network with self‑attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuanvolutionFilter(nn.Module):
    """2×2 convolutional filter that downsamples the input image."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)  # flatten per batch


class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block that operates on the flattened feature vector."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return torch.bmm(scores, v)


class QuanvolutionClassifier(nn.Module):
    """Hybrid network: quanvolution → self‑attention → linear head."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.attention = ClassicalSelfAttention(embed_dim=4 * 14 * 14)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)          # shape: (B, 4*14*14)
        attn_out = self.attention(features) # shape: (B, 4*14*14)
        logits = self.linear(attn_out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier", "ClassicalSelfAttention"]
