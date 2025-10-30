"""Hybrid QCNN model combining classical self‑attention with a QCNN‑style backbone.

The module defines a PyTorch model that first applies a learnable
self‑attention block to the input, then feeds the result through a
stack of fully‑connected layers that mirror the original QCNN
architecture.  The attention mechanism captures long‑range
dependencies, while the QCNN backbone compresses the representation
in a hierarchical manner.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Learnable self‑attention block."""

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable linear maps for query and key
        self.query_map = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_map = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Tensor of the same shape after attention.
        """
        query = self.query_map(inputs)  # (batch, seq_len, embed_dim)
        key = self.key_map(inputs)      # (batch, seq_len, embed_dim)
        scores = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim),
            dim=-1,
        )
        return torch.matmul(scores, inputs)

class HybridQCNNModel(nn.Module):
    """Hybrid QCNN that applies self‑attention before the QCNN feature extractor."""

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        # QCNN feature extractor (same topology as the seed)
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Tensor of shape (batch, 1) with sigmoid activation.
        """
        # Apply self‑attention
        attn_out = self.attention(inputs)  # (batch, seq_len, embed_dim)
        # Collapse sequence dimension for the QCNN backbone
        x = attn_out.view(attn_out.size(0), -1)  # (batch, seq_len*embed_dim)
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def HybridQCNN() -> HybridQCNNModel:
    """Factory returning the configured hybrid QCNN model."""
    return HybridQCNNModel()

__all__ = ["HybridQCNN", "HybridQCNNModel", "ClassicalSelfAttention"]
