"""Hybrid quanvolution + transformer model with kernel regularization (classical)."""

from __future__ import annotations

import numpy as np
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalQuanvolutionFilter(nn.Module):
    """2‑D convolutional filter that mimics a quantum kernel on image patches."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)


class ClassicalTransformerBlock(nn.Module):
    """Standard transformer block using PyTorch primitives."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class ClassicalKernel(nn.Module):
    """Radial Basis Function kernel for feature similarity."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class EstimatorHead(nn.Module):
    """Simple feed‑forward regressor."""
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuanvolutionHybridModel(nn.Module):
    """Hybrid architecture combining quanvolution, transformer, and kernel modules."""
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_blocks: int = 2,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter(in_channels, embed_dim // 2)
        self.transformer = nn.Sequential(
            *[ClassicalTransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
        )
        self.kernel = ClassicalKernel(kernel_gamma)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.estimator = EstimatorHead(embed_dim)

    def forward(self, x: torch.Tensor, mode: str = "classify") -> torch.Tensor:
        """`mode` can be 'classify' or'regress'."""
        features = self.qfilter(x)
        seq_len = features.size(1) // 64
        feats = features.view(x.size(0), seq_len, 64)
        feats = self.transformer(feats)
        pooled = feats.mean(dim=1)
        if mode == "classify":
            return F.log_softmax(self.classifier(pooled), dim=-1)
        return self.estimator(pooled)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])


__all__ = ["QuanvolutionHybridModel"]
