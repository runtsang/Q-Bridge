"""Unified classical regression model with a quantum-inspired confidence head.

This module fuses the classical regression seed with the hybrid binary
classifier head.  It learns a fast surrogate model and a confidence
estimate that mimics the quantum expectation head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple

def generate_augmented_superposition_data(
    num_features: int,
    samples: int,
    noise_std: float = 0.05,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += rng.normal(scale=noise_std, size=y.shape).astype(np.float32)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.05, seed: int | None = None):
        self.features, self.labels = generate_augmented_superposition_data(num_features, samples, noise_std, seed)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class UnifiedQuantumRegression(nn.Module):
    """Classical regression model with a quantum‑inspired confidence head."""
    def __init__(self, num_features: int, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(states)
        pred = self.reg_head(feats).squeeze(-1)
        var = self.conf_head(feats).squeeze(-1)
        return pred, var

__all__ = ["UnifiedQuantumRegression", "RegressionDataset", "generate_augmented_superposition_data"]
