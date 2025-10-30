"""Hybrid classical QCNN with attention for 8‑bit inputs."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

class QCNNGen320(nn.Module):
    """
    Convolution‑pooling network augmented with a self‑attention block.

    The architecture mimics a QCNN stack but replaces the final linear head
    with a multi‑head self‑attention that learns feature interactions
    before the classification layer.
    """
    def __init__(self, embed_dim: int = 4, num_heads: int = 2) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution‑pooling stack
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Self‑attention over the single feature vector
        seq = x.unsqueeze(1)  # (batch, seq_len=1, embed_dim)
        attn_output, _ = self.attention(seq, seq, seq)
        attn_output = attn_output.squeeze(1)

        return torch.sigmoid(self.head(attn_output))

def QCNNGen320() -> QCNNGen320:
    """Factory returning a pre‑configured QCNN‑attention model."""
    return QCNNGen320()

__all__ = ["QCNNGen320", "QCNNGen320"]
