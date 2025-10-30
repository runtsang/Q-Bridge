"""Combined QCNN and self‑attention model for classical training."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Drop‑in replacement for the original SelfAttention helper."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable linear projections
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply scaled dot‑product attention."""
        query = self.query_proj(inputs)
        key   = self.key_proj(inputs)
        value = self.value_proj(inputs)
        scores = torch.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

class QCNNWithAttentionModel(nn.Module):
    """QCNN‑style network augmented with a self‑attention block."""
    def __init__(self, input_dim: int = 8, embed_dim: int = 4) -> None:
        super().__init__()
        # Feature map and convolutional stages (mirroring the QCNN seed)
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Self‑attention layer
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        # Output head
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Attention operates on the intermediate representation
        x = self.attention(x)
        return torch.sigmoid(self.head(x))

def QCNNWithAttention() -> QCNNWithAttentionModel:
    """Factory returning a pre‑configured QCNN‑with‑attention model."""
    return QCNNWithAttentionModel()

__all__ = ["QCNNWithAttention", "QCNNWithAttentionModel"]
