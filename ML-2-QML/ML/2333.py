"""Combined classical regression with self‑attention feature extractor."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data using a superposition‑inspired target function."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class SelfAttentionLayer(nn.Module):
    """Classical self‑attention module used as a feature extractor."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)


class RegressionDataset(Dataset):
    """Dataset yielding classical feature vectors and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QuantumRegressionSelfAttentionML(nn.Module):
    """Classical regression model that first applies self‑attention."""
    def __init__(self, num_features: int, attention_dim: int = 32):
        super().__init__()
        self.attention = SelfAttentionLayer(attention_dim)
        self.proj = nn.Linear(attention_dim, num_features)
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x)
        proj_out = self.proj(attn_out)
        return self.net(proj_out).squeeze(-1)


__all__ = ["QuantumRegressionSelfAttentionML", "RegressionDataset", "generate_superposition_data"]
