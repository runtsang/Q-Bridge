from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ClassicalSelfAttention:
    """Simple self‑attention wrapper that mimics the behaviour of the quantum attention block."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        # learnable linear projections
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, embed_dim)
        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of the same shape.
        """
        query = self.query_proj(inputs)
        key = self.key_proj(inputs)
        scores = torch.softmax(query @ key.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

class QCNNWithAttention(nn.Module):
    """
    Classical QCNN architecture augmented with a self‑attention module.
    The attention is applied to the output of the final convolutional block before the classification head.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        # Feature extraction layers
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Attention module
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        # Classification head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Apply attention
        x = self.attention(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNWithAttention:
    """Factory returning a QCNN model enhanced with self‑attention."""
    return QCNNWithAttention()

__all__ = ["QCNN", "QCNNWithAttention"]
