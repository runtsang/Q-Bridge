"""Hybrid classical regression model with self‑attention."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic data: y = sin(sum(x)) + 0.1*cos(2*sum(x))."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping synthetic regression data."""
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
    """Simple self‑attention layer using trainable rotation and entangle matrices."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Parameters for rotations and entanglements
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = torch.matmul(inputs, self.rotation)
        key = torch.matmul(inputs, self.entangle)
        value = inputs
        scores = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, value)

class HybridRegressionModel(nn.Module):
    """Classical regression model with self‑attention followed by a feed‑forward head."""
    def __init__(self, num_features: int, attention_dim: int = 8):
        super().__init__()
        self.attention = ClassicalSelfAttention(attention_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(attention_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # Linear layer to map input features to attention dimension
        self.project = nn.Linear(num_features, attention_dim)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Project to attention space
        proj = self.project(state_batch)
        # Apply self‑attention
        attn_out = self.attention(proj)
        # Feed‑forward head
        return self.feedforward(attn_out).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
