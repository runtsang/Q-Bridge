"""Hybrid classical neural network combining feed‑forward, convolution‑like,
and self‑attention layers."""
from __future__ import annotations

import math
import torch
from torch import nn
import numpy as np


class SelfAttentionModule(nn.Module):
    """Simple dot‑product self‑attention with learnable linear maps."""

    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.query.out_features), dim=-1)
        return scores @ v


class HybridEstimatorQNN(nn.Module):
    """Hybrid classical neural network combining feed‑forward, convolution‑like,
    and self‑attention layers."""
    def __init__(self) -> None:
        super().__init__()
        # Feature map (like EstimatorQNN)
        self.feature_map = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
        )
        # Convolution‑style stack (like QCNNModel)
        self.conv1 = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Self‑attention
        self.attention = SelfAttentionModule(embed_dim=4)
        # Final head
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.attention(x)
        out = torch.sigmoid(self.head(x))
        return out


def EstimatorQNN() -> HybridEstimatorQNN:
    """Factory returning the configured :class:`HybridEstimatorQNN`."""
    return HybridEstimatorQNN()


__all__ = ["EstimatorQNN"]
