"""Extended classical QCNN with residuals, dropout and layer normalisation."""
from __future__ import annotations

import torch
from torch import nn, Tensor


class _ResidualBlock(nn.Module):
    """A lightweight residual block that keeps dimensions unchanged."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.fc(x) + x))


class QCNN(nn.Module):
    """
    A stack of fully‑connected layers that mimics a QCNN but with
    modern regularisation techniques.

    Architecture
    ------------
    * feature_map  : Linear(8 -> 32) → Tanh
    * conv1        : Linear(32 -> 32) → Tanh
    * res1         : ResidualBlock(32)
    * pool1        : Linear(32 -> 24) → Tanh
    * conv2        : Linear(24 -> 16) → Tanh
    * res2         : ResidualBlock(16)
    * pool2        : Linear(16 -> 8) → Tanh
    * conv3        : Linear(8  -> 8)  → Tanh
    * dropout      : Dropout(0.3)
    * head         : Linear(8 -> 1)
    """
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 32), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(32, 32), nn.Tanh())
        self.res1 = _ResidualBlock(32)
        self.pool1 = nn.Sequential(nn.Linear(32, 24), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(24, 16), nn.Tanh())
        self.res2 = _ResidualBlock(16)
        self.pool2 = nn.Sequential(nn.Linear(16, 8), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(8, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNN:
    """
    Factory returning an instance of the extended QCNN.
    """
    return QCNN()


__all__ = ["QCNN", "QCNNModel"]
