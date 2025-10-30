"""Hybrid classical QCNN with self‑attention.

This class mirrors the structure of the original QCNNModel but
augments the feature extraction pipeline with two classical
self‑attention blocks.  The design follows the pattern of the
`SelfAttention` helper while preserving the simple linear
convolution‑pooling skeleton.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np


class ClassicalSelfAttention(nn.Module):
    """Self‑attention module mirroring the original quantum interface."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Linear projections for query and key
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query(x)
        key = self.key(x)
        scores = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim),
            dim=-1,
        )
        return torch.matmul(scores, x)


class QCNNGen106(nn.Module):
    """Classical QCNN augmented with self‑attention layers."""

    def __init__(self, input_dim: int = 8, embed_dim: int = 4) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.attn1 = ClassicalSelfAttention(embed_dim=embed_dim)
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.attn2 = ClassicalSelfAttention(embed_dim=embed_dim)
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.attn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.attn2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


__all__ = ["QCNNGen106"]
