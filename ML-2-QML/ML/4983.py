"""QCNN__gen244.py – Classical hybrid model."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np


class SelfAttentionLayer(nn.Module):
    """Simple self‑attention block built from linear projections."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class QCNN__gen244(nn.Module):
    """Hybrid classical network combining convolutional, self‑attention, and FC layers."""

    def __init__(self) -> None:
        super().__init__()
        # Convolution‑style feature extractor
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Self‑attention block
        self.attention = SelfAttentionLayer(embed_dim=4)

        # Fully connected head
        self.fc = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.attention(x)
        out = self.fc(x)
        return torch.sigmoid(out)


def QCNN__gen244() -> QCNN__gen244:
    """Factory returning the configured :class:`QCNN__gen244`."""
    return QCNN__gen244()


__all__ = ["QCNN__gen244", "QCNN__gen244"]
