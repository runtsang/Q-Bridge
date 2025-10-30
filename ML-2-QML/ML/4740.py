"""
Hybrid classical estimator combining feed‑forward, self‑attention and convolutional
layers.  The architecture mirrors the classical `EstimatorQNN` seed, extends it with
a learnable self‑attention block and a stack of convolution–pooling layers inspired
by `QCNN`.  The network can be trained with any standard PyTorch optimizer.

The class is intentionally lightweight to keep training fast while still
demonstrating the synergy between attention, convolution and fully‑connected
components.
"""

from __future__ import annotations

import math
import torch
from torch import nn


class SelfAttention(nn.Module):
    """
    Learnable multi‑head self‑attention with a single head.
    Uses ``nn.Linear`` to construct query, key and value projections.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Attention output of shape ``(batch, embed_dim)``.
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class HybridEstimatorQNN(nn.Module):
    """
    Classical hybrid estimator.

    Architecture
    ------------
    * ``feature`` : Linear(2 → 8) + Tanh
    * ``attn``    : SelfAttention(8)
    * ``conv1``   : Linear(8 → 16) + Tanh
    * ``pool1``   : Linear(16 → 12) + Tanh
    * ``conv2``   : Linear(12 → 8) + Tanh
    * ``pool2``   : Linear(8 → 4) + Tanh
    * ``conv3``   : Linear(4 → 4) + Tanh
    * ``head``    : Linear(4 → 1) + Sigmoid
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(2, 8), nn.Tanh())
        self.attn = SelfAttention(embed_dim=8)
        self.conv1 = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Regression output of shape ``(batch, 1)``.
        """
        x = self.feature(x)
        attn_out = self.attn(x)
        x = x + attn_out  # residual connection
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def HybridEstimatorQNN() -> HybridEstimatorQNN:
    """
    Factory that returns an instance of the hybrid classical estimator.
    """
    return HybridEstimatorQNN()


__all__ = ["HybridEstimatorQNN"]
