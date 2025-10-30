from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class QCNNGen333(nn.Module):
    """Hybrid QCNN class combining classical convolution, pooling, attention, and sampler."""
    def __init__(self) -> None:
        super().__init__()
        # Feature map analogous to a quantum feature map
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        # Convolution and pooling layers
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Self‑attention learnable parameters
        self.attn_rot = nn.Parameter(torch.randn(4, 4))
        self.attn_ent = nn.Parameter(torch.randn(4, 4))
        # Sampler network
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )
        # Classification head
        self.head = nn.Linear(6, 1)

    def _attention(self, inputs: torch.Tensor) -> torch.Tensor:
        """Classical self‑attention block."""
        query = inputs @ self.attn_rot
        key = inputs @ self.attn_ent
        scores = F.softmax(query @ key.T / np.sqrt(query.size(-1)), dim=-1)
        return scores @ inputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Apply self‑attention
        attn_out = self._attention(x)
        # Sampler: use first two features
        samp_out = F.softmax(self.sampler(x[:, :2]), dim=-1)
        # Concatenate and classify
        combined = torch.cat([attn_out, samp_out], dim=-1)
        return torch.sigmoid(self.head(combined))

def QCNN() -> QCNNGen333:
    """Factory returning a configured QCNNGen333 instance."""
    return QCNNGen333()

__all__ = ["QCNN", "QCNNGen333"]
