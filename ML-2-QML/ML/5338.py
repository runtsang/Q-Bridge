"""Hybrid classical model combining CNN, transformer, sampler and optional kernel."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

# Feature extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_ch: int = 1, out_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(16 * 7 * 7, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))

# Positional encoding
class PositionalEncoder(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(att))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# Sampler head
class SamplerHead(nn.Module):
    def __init__(self, dim: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4),
            nn.Tanh(),
            nn.Linear(4, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

# Kernel utilities
class RBFKernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    k = RBFKernel(gamma)
    return np.array([[k(x, y).item() for y in b] for x in a])

# Hybrid model
class HybridNATModel(nn.Module):
    """Classical hybrid model combining CNN, transformer, sampler and optional kernel."""
    def __init__(
        self,
        in_ch: int = 1,
        cnn_dim: int = 64,
        embed_dim: int = 64,
        heads: int = 4,
        ffn_dim: int = 128,
        blocks: int = 2,
        n_classes: int = 4,
        use_kernel: bool = False,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.extractor = CNNFeatureExtractor(in_ch, cnn_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlock(embed_dim, heads, ffn_dim) for _ in range(blocks)]
        )
        self.head = SamplerHead(embed_dim, n_classes)
        self.use_kernel = use_kernel
        if use_kernel:
            self.prototypes = nn.Parameter(torch.randn(n_classes, embed_dim))
            self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extractor(x)  # [B, C]
        tokens = feats.unsqueeze(1)  # [B, 1, C]
        tokens = self.pos_enc(tokens)
        out = self.transformers(tokens).squeeze(1)
        logits = self.head(out)
        if self.use_kernel:
            sims = torch.exp(-self.gamma * torch.cdist(out.unsqueeze(1), self.prototypes.unsqueeze(0)) ** 2)
            return logits, sims
        return logits

__all__ = ["HybridNATModel"]
