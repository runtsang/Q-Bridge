"""Hybrid self‑attention regression – classical implementation.

The class combines a lightweight multi‑head self‑attention extractor
with a fully‑connected regression head.  It re‑uses the data
generation from the QuantumRegression seed but operates entirely on
CPU/GPU tensors.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics a quantum superposition pattern.
    Each sample is drawn uniformly from [-1, 1] and a target is produced
    by a smooth non‑linear mapping.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset wrapping the synthetic superposition data.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class ClassicalSelfAttention(nn.Module):
    """
    Simple dot‑product self‑attention block with learnable linear projections.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, embed_dim)
        """
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


class HybridSelfAttentionRegression(nn.Module):
    """
    Classical hybrid model: self‑attention feature extractor + regression head.
    """
    def __init__(self, num_features: int, embed_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim, num_heads)
        self.input_proj = nn.Linear(num_features, embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : Tensor of shape (batch, seq_len, num_features)
        """
        x = self.input_proj(states)
        attn_out = self.attention(x)
        # pool over sequence dimension
        pooled = attn_out.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


__all__ = ["HybridSelfAttentionRegression", "RegressionDataset", "generate_superposition_data"]
