"""Hybrid self‑attention module combining QCNN feature extraction and classical attention."""

from __future__ import annotations

import math
import torch
from torch import nn

class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating the quantum convolution steps."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridSelfAttention(nn.Module):
    """Classical hybrid self‑attention that first extracts QCNN‑style features
    and then applies a multi‑head attention over the feature dimensions.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qcnn = QCNNModel()
        # Linear projections for query, key and value on the 4‑dimensional QCNN output
        self.q_linear = nn.Linear(4, 4)
        self.k_linear = nn.Linear(4, 4)
        self.v_linear = nn.Linear(4, 4)
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, 8) – the same dimensionality used by the QCNN.
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1) after self‑attention and a sigmoid activation.
        """
        features = self.qcnn(inputs)  # (batch, 4)
        q = self.q_linear(features)   # (batch, 4)
        k = self.k_linear(features)   # (batch, 4)
        v = self.v_linear(features)   # (batch, 4)

        d_k = q.size(-1)
        scores = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d_k), dim=-1)  # (batch, 4, 4)
        weighted = torch.einsum('bij,bj->bi', scores, v)  # (batch, 4)
        out = torch.sigmoid(self.head(weighted))          # (batch, 1)
        return out

def SelfAttention() -> HybridSelfAttention:
    """Factory returning the configured :class:`HybridSelfAttention`."""
    return HybridSelfAttention()

__all__ = ["SelfAttention", "HybridSelfAttention", "QCNNModel"]
