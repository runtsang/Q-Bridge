"""Hybrid self‑attention + regression model – classical implementation.

The module defines a dataset generator, a classical self‑attention block,
and a lightweight regression network that consumes the attention outputs.
All components expose a simple ``run`` method so they can be swapped
in a larger training loop without modification.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics a superposition‑like structure.
    Each sample is a random vector in [-1, 1]^num_features.
    Labels are a smooth nonlinear function of the sum of features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Torch dataset wrapping the synthetic superposition data.
    Returns a dict with ``states`` and ``target`` tensors.
    """
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
    """
    Learnable self‑attention layer that projects inputs into query/key/value
    spaces using trainable linear maps and computes attention scores.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(inputs)          # (B, D)
        k = self.k_proj(inputs)          # (B, D)
        v = self.v_proj(inputs)          # (B, D)
        scores = torch.softmax((q @ k.t()) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class HybridSelfAttentionRegression(nn.Module):
    """
    Classic hybrid model that first applies self‑attention to the input
    features and then feeds the attended representation into a small MLP.
    """
    def __init__(self, num_features: int, embed_dim: int = 32):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim=num_features)
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(state_batch)
        return self.regressor(attn_out).squeeze(-1)

    def run(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Alias for forward to match the quantum interface.
        """
        return self.forward(state_batch)


__all__ = [
    "HybridSelfAttentionRegression",
    "RegressionDataset",
    "generate_superposition_data",
    "ClassicalSelfAttention",
]
