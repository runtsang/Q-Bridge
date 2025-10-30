"""Hybrid QCNN + Self‑Attention model for classical training."""

from __future__ import annotations

import math
import torch
from torch import nn
import numpy as np


class QCNNSelfAttentionModel(nn.Module):
    """
    Convolution‑inspired fully‑connected layers followed by a learnable
    self‑attention block.  The attention is implemented in a way that
    mirrors the quantum version: rotation and entangle parameters are
    learned separately and combined via a soft‑max.
    """

    def __init__(self, input_dim: int = 8, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Convolution‑like linear layers
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Attention parameters (learnable)
        self.rot_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.ent_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

        # Output head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution‑pooling stages
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Self‑attention block
        query = x @ self.rot_params
        key = x @ self.ent_params
        scores = torch.softmax(query @ key.T / math.sqrt(self.embed_dim), dim=-1)
        value = x
        attn_out = scores @ value

        # Combine and classify
        combined = x + attn_out
        logits = self.head(combined)
        return torch.sigmoid(logits)


def QCNN() -> QCNNSelfAttentionModel:
    """Factory returning a ready‑to‑use QCNN‑SelfAttention model."""
    return QCNNSelfAttentionModel()


__all__ = ["QCNN", "QCNNSelfAttentionModel"]
