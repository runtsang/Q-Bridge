"""
Hybrid model combining classical convolution, self‑attention, and fraud‑detection style layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalSelfAttention(nn.Module):
    """
    Simple attention module mimicking quantum self‑attention.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = np.sqrt(embed_dim)

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.out(out)


class HybridQuanvolutionModel(nn.Module):
    """
    Classical hybrid model that:
    * uses a 2×2 convolutional filter (inspired by Quanvolution)
    * applies a classical self‑attention block
    * passes the result through fraud‑detection style dense layers
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 2×2 convolution down‑sampling
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

        # attention on the 196 patches (28×28 / 2)
        self.attn = ClassicalSelfAttention(embed_dim=4)

        # fraud‑detection style dense layers
        self.fc1 = nn.Linear(4 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 28, 28)
        """
        # Convolutional feature extraction
        feat = self.conv(x)                     # (batch, 4, 14, 14)
        seq = feat.view(feat.size(0), 196, 4)    # (batch, seq_len, embed_dim)

        # Self‑attention
        attn_feat = self.attn(seq)
        flat = attn_feat.view(feat.size(0), -1)  # (batch, 4*14*14)

        # Dense layers
        h = F.relu(self.fc1(flat))
        h = F.relu(self.fc2(h))
        logits = self.out(h)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolutionModel"]
