"""Hybrid classical regression model with self‑attention preprocessing."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with a sinusoidal relationship."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block mirroring the quantum interface."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable linear projections for query/key/value
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q = self.to_q(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        # scaled dot‑product attention
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class HybridRegressor(nn.Module):
    """Classical regression pipeline: attention ➜ feed‑forward."""
    def __init__(self, num_features: int, embed_dim: int = 32):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.projection = nn.Linear(num_features, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Map raw features to embedding space
        x = self.projection(state_batch)
        # Apply self‑attention
        attn_out = self.attention(x)
        # Predict target
        return self.net(attn_out).squeeze(-1)

__all__ = ["HybridRegressor", "RegressionDataset", "generate_superposition_data"]
