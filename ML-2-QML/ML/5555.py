"""HybridQCNN – classical implementation

This module replicates the convolutional workflow of the original QCNN and augments it with
classical self‑attention and a lightweight softmax sampler.  The network is fully
trainable with PyTorch and can be dropped into any standard training loop.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np


class ClassicalSelfAttention(nn.Module):
    """Simple dot‑product attention implemented in PyTorch."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # learnable linear projections for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of shape (batch, seq_len, embed_dim).
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scores = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ V


class SamplerModule(nn.Module):
    """Softmax sampler that mimics the Qiskit SamplerQNN output."""

    def __init__(self, input_dim: int, hidden_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        return torch.softmax(self.net(x), dim=-1)


class HybridQCNN(nn.Module):
    """Classical hybrid QCNN that combines convolution, attention and sampling."""

    def __init__(self, input_dim: int = 8, embed_dim: int = 4) -> None:
        super().__init__()
        # Convolution‑inspired feature extractor
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Classical self‑attention
        self.attention = ClassicalSelfAttention(embed_dim)
        # Sampler
        self.sampler = SamplerModule(input_dim=4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Probabilities of shape (batch, 2).
        """
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # reshape for attention: (batch, seq_len=1, embed_dim)
        x = x.unsqueeze(1)
        x = self.attention(x).squeeze(1)
        return self.sampler(x)


def HybridQCNNFactory() -> HybridQCNN:
    """Convenience factory returning a ready‑to‑train instance."""
    return HybridQCNN()


__all__ = ["HybridQCNN", "HybridQCNNFactory", "ClassicalSelfAttention", "SamplerModule"]
