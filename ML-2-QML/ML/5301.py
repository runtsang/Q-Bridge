from __future__ import annotations
import torch
from torch import nn
import numpy as np

class SelfAttentionModule(nn.Module):
    """
    Lightweight self‑attention block that mirrors the quantum
    attention circuit.  It uses linear projections to obtain
    query/key/value and a soft‑max over their dot product.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        scores = torch.softmax((q @ k.T) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class HybridQuantumLayer(nn.Module):
    """
    Classical analogue of a QCNN with an added attention residual.
    The architecture is inspired by the four reference pairs:
      * Fully connected layer (FCL)
      * QCNN feature map & convolution/pooling blocks
      * Self‑attention mechanism
      * Fraud‑detection style layer (bias + activation)
    """
    def __init__(self, input_dim: int = 8, embed_dim: int = 4):
        super().__init__()
        # Feature map and convolution/pooling stack
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Attention residual
        self.attention = SelfAttentionModule(embed_dim)
        # Final head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        att = self.attention(x)
        x = x + att  # residual fusion
        return torch.sigmoid(self.head(x))

__all__ = ["HybridQuantumLayer", "SelfAttentionModule"]
