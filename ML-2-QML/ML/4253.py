"""Unified classical sampler network that combines convolutional layers and self‑attention.

The architecture mirrors the QCNN but augments it with a trainable attention block.
The final output is a single probability (sigmoid) suitable for binary classification."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedSamplerQNN(nn.Module):
    """
    Feature extractor -> convolutional layers -> self‑attention -> classification head.
    """
    def __init__(self, input_dim: int = 8, embed_dim: int = 4) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh(),
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.Tanh(),
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.Tanh(),
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
        )
        # Attention parameters
        self.attn_rot = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.attn_ent = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.head = nn.Linear(4, 1)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a self‑attention block using learnable rotation and entanglement matrices.
        """
        # x: (batch, embed_dim)
        query = x @ self.attn_rot
        key = x @ self.attn_ent
        scores = F.softmax(query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)), dim=-1)
        return scores @ x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature map, conv/pool layers, attention, and head.
        """
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Attention
        x = self._attention(x)
        out = torch.sigmoid(self.head(x))
        return out

def SamplerQNN() -> UnifiedSamplerQNN:
    """Factory returning a fully‑configured :class:`UnifiedSamplerQNN`."""
    return UnifiedSamplerQNN()
