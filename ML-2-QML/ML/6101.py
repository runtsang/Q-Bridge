from __future__ import annotations
import numpy as np
import torch
from torch import nn
from typing import Tuple

class SelfAttentionLayer(nn.Module):
    """Scaled dot‑product self‑attention with learnable linear projections."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        return torch.matmul(scores, V)

class HybridSelfAttentionQCNN(nn.Module):
    """
    Classical network that fuses a convolution‑like feature extractor (QCNN style)
    with a self‑attention module.  The architecture mirrors the original
    SelfAttention and QCNN seeds but interleaves attention after the first
    convolutional block to capture long‑range correlations before pooling.
    """
    def __init__(self, in_features: int = 8, embed_dim: int = 16):
        super().__init__()
        # Feature map inspired by QCNN
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.Tanh()
        )
        # Convolution‑like layers
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8),  nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4),   nn.Tanh())
        # Pooling layers
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4),  nn.Tanh())
        # Self‑attention bridge
        self.attention = SelfAttentionLayer(embed_dim=16)
        # Final classifier
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.attention(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def SelfAttention() -> HybridSelfAttentionQCNN:
    """Factory returning a hybrid classical self‑attention / QCNN network."""
    return HybridSelfAttentionQCNN()
