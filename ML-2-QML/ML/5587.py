"""Hybrid kernel model blending classical RBF, CNN, self‑attention, and fraud‑detection scaling."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax
from typing import Sequence

class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block used in the classical hybrid."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class FraudScalingLayer(nn.Module):
    """Linear + activation + scaling layer mirroring the fraud‑detection photonic network."""

    def __init__(self, in_features: int, out_features: int, scale: float = 1.0, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()
        self.scale = scale
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        return out * self.scale + self.shift

class HybridKernelModel(nn.Module):
    """Classical hybrid kernel that combines a CNN frontend, self‑attention, fraud‑style scaling,
    and an RBF kernel on the resulting feature vectors."""

    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        # Feature extractor: shallow CNN
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Project flattened features to the attention dimension
        self.proj = nn.Linear(16 * 7 * 7, embed_dim)
        # Self‑attention
        self.attention = ClassicalSelfAttention(embed_dim)
        # Fraud‑style scaling
        self.fraud = FraudScalingLayer(
            in_features=embed_dim,
            out_features=embed_dim,
            scale=1.0,
            shift=0.0,
        )
        self.gamma = 1.0

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Encode both inputs
        def encode(z: torch.Tensor) -> torch.Tensor:
            feat = self.features(z)
            flat = feat.view(z.shape[0], -1)
            proj = self.proj(flat)
            att = self.attention(proj)
            return self.fraud(att)

        fx = encode(x)
        fy = encode(y)
        diff = fx - fy
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True)).squeeze()

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
        model = HybridKernelModel()
        model.gamma = gamma
        return np.array([[model(a_i, b_j).item() for b_j in b] for a_i in a])

__all__ = ["HybridKernelModel"]
