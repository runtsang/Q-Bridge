"""Unified classical estimator combining regression, self‑attention, and batch normalization.

The model mirrors the original EstimatorQNN but augments it with a trainable
self‑attention module and additional normalization layers.  It can be used
as a drop‑in replacement for the baseline model while providing richer
representations.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    """Trainable self‑attention layer for 2‑dimensional inputs.

    It projects the input into an ``embed_dim``‑dimensional space using
    separate linear maps for query, key and value, computes scaled dot‑product
    attention, and returns the attended representation.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(2, embed_dim, bias=False)
        self.key = nn.Linear(2, embed_dim, bias=False)
        self.value = nn.Linear(2, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class EstimatorQNNGen157(nn.Module):
    """Hybrid feed‑forward regressor with self‑attention and batch‑norm.

    The architecture is:
        1) Self‑attention block (2 → 4)
        2) Concatenation of original features and attention output (6 → 8)
        3) Two fully‑connected layers with batch‑norm and non‑linearities
        4) Final regression head
    """
    def __init__(self) -> None:
        super().__init__()
        self.attention = SelfAttentionModule(embed_dim=4)
        self.fc1 = nn.Linear(6, 8)
        self.bn1 = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, 4)
        self.bn2 = nn.BatchNorm1d(4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, 2)
        attn = self.attention(x)            # (batch, 4)
        combined = torch.cat([x, attn], dim=-1)  # (batch, 6)
        out = self.fc1(combined)
        out = self.bn1(out)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        return out


__all__ = ["EstimatorQNNGen157"]
