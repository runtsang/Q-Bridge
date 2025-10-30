"""Hybrid classical regression model with self‑attention feature extraction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SelfAttention(nn.Module):
    """Trainable self‑attention module mirroring the quantum interface."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.key_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = torch.matmul(inputs, self.query_params)
        key = torch.matmul(inputs, self.key_params)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

class QModel(nn.Module):
    """Hybrid regression model using a self‑attention feature extractor followed by an MLP."""
    def __init__(self, num_features: int, embed_dim: int | None = None):
        super().__init__()
        if embed_dim is None:
            embed_dim = num_features
        self.attention = SelfAttention(embed_dim=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attention(state_batch)
        return self.mlp(attn_out).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
