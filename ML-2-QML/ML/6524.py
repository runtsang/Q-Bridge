"""Hybrid classical self‑attention regression model.

This module builds on the SelfAttention and QuantumRegression seeds,
providing a pure‑Python implementation suitable for CPU/GPU training.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation – identical to the quantum regression seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    Samples are drawn uniformly from [-1, 1] and the target is a smooth,
    nonlinear function of the sum of the features.  The same function
    is used in the quantum seed to enable head‑to‑head comparisons.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset wrapper
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and scalar targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Classical self‑attention block
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """A lightweight multi‑head self‑attention layer."""
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape  # batch, seq_len, embed_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        attn_weights = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(D // self.num_heads), dim=-1)
        attn_output = (attn_weights @ v).reshape(B, N, D)
        return self.out_proj(attn_output)

# --------------------------------------------------------------------------- #
# Hybrid classical regression model
# --------------------------------------------------------------------------- #
class HybridSelfAttentionRegression(nn.Module):
    """Classical regression model with a self‑attention feature extractor."""
    def __init__(self, num_features: int, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(num_features, embed_dim)
        self.attention = ClassicalSelfAttention(embed_dim, num_heads)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(state_batch)
        # reshape for seq_len=1 to feed through attention
        x = x.unsqueeze(1)
        attn_out = self.attention(x).squeeze(1)
        return self.reg_head(attn_out).squeeze(-1)

__all__ = ["HybridSelfAttentionRegression", "RegressionDataset", "generate_superposition_data"]
