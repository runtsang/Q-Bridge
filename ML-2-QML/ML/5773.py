"""Hybrid QCNN with classical self‑attention layers."""

import torch
from torch import nn
import numpy as np


class SelfAttentionBlock(nn.Module):
    """Classical self‑attention block with residual connection."""
    def __init__(self, embed_dim: int, heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.scale = embed_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, seq_len, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out) + x  # residual connection


class QCNNAttention(nn.Module):
    """QCNN architecture augmented with two self‑attention blocks."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.attn1 = SelfAttentionBlock(embed_dim=16)
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.attn2 = SelfAttentionBlock(embed_dim=8)
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        # reshape to (batch, seq_len=1, embed_dim)
        x = x.unsqueeze(1)
        x = self.attn1(x).squeeze(1)
        x = self.pool1(x)
        x = self.conv2(x)
        x = x.unsqueeze(1)
        x = self.attn2(x).squeeze(1)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNAttention:
    """Factory returning the hybrid QCNN‑Attention model."""
    return QCNNAttention()


__all__ = ["QCNN", "QCNNAttention", "SelfAttentionBlock"]
