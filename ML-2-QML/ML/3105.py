"""Hybrid classical regression model with self‑attention.

This module extends the original `QuantumRegression.py` by adding a
self‑attention block after the hidden layers.  The attention weights
are learned via a small linear layer followed by a softmax, allowing
the model to focus on the most informative features of the quantum‑like
representation.  The dataset and data‑generation logic are identical
to the seed, ensuring reproducibility.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data using a polar‑angle superposition.

    The returned states are real‑valued vectors of dimension ``num_features``.
    Labels are ``sin(2θ) * cos(φ)`` where θ and φ are random angles.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    # Construct a simple real‑valued representation (e.g., sinθ, cosθ)
    states = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    # If more features are requested, pad with zeros
    if num_features > 2:
        states = np.concatenate([states, np.zeros((samples, num_features - 2))], axis=1)
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.float32), labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic data."""
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
    """Simple learnable attention over the hidden representation."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = nn.Linear(embed_dim, embed_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = self.softmax(self.attn(features))
        return features * weights

class QModel(nn.Module):
    """Hybrid classical regression model with an attention head."""
    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = nn.Linear(num_features, 32)
        self.hidden = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.attention = ClassicalSelfAttention(embed_dim=16)
        self.head = nn.Linear(16, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.encoder(state_batch)
        x = self.hidden(x)
        x = self.attention(x)
        return self.head(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
