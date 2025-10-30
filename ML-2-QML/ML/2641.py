"""Hybrid classical regression model with self‑attention and quantum‑style feature extraction.

The module defines:
* generate_superposition_data – synthetic dataset generator.
* RegressionDataset – PyTorch Dataset returning states and targets.
* SelfAttention – lightweight self‑attention block.
* HybridRegressionModel – classical neural network that first applies self‑attention
  and then a feed‑forward network.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    Features are uniformly sampled in [-1, 1] and the target is a nonlinear
    function of the sum of the features, mimicking a superposition state.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that yields feature vectors and scalar targets."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class SelfAttention:
    """Simple self‑attention block used before the feed‑forward network."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        # Trainable linear projections for query, key and value
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply self‑attention to the input batch."""
        q = self.w_q(inputs)
        k = self.w_k(inputs)
        v = self.w_v(inputs)
        scores = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class HybridRegressionModel(nn.Module):
    """Classical regression model that first applies self‑attention."""

    def __init__(self, num_features: int):
        super().__init__()
        self.attention = SelfAttention(num_features)
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attention.forward(state_batch)
        return self.net(attn_out).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
