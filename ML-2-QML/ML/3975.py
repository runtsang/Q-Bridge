import torch
from torch import nn
import numpy as np


class SelfAttentionModule(nn.Module):
    """Learnable self‑attention block used in the classical hybrid QCNN."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = torch.matmul(x, self.rotation)
        key = torch.matmul(x, self.entangle)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, x)


class QCNNSelfAttentionHybrid(nn.Module):
    """Classical QCNN architecture augmented with a self‑attention layer."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.attention = SelfAttentionModule(embed_dim=4)
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.attention(x)
        return torch.sigmoid(self.head(x))


def QCNNHybrid() -> QCNNSelfAttentionHybrid:
    """Factory returning a configured QCNNSelfAttentionHybrid instance."""
    return QCNNSelfAttentionHybrid()


__all__ = ["QCNNHybrid", "QCNNSelfAttentionHybrid"]
