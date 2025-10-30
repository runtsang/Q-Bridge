"""Enhanced classical regression model with feature‑importance gating."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data mimicking a superposition of two basis states.
    The output labels are a noisy sinusoidal function of the sum of inputs.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    # add small Gaussian noise to labels
    y += 0.02 * np.random.randn(samples).astype(np.float32)
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

class QModel(nn.Module):
    """
    Classical neural network with a learnable feature‑importance gate.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64):
        super().__init__()
        # Gate that learns to weight each input feature
        self.feature_gate = nn.Linear(num_features, num_features)
        # Encoder that extracts higher‑level representations
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Final regression head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Learn a soft mask for each feature
        mask = torch.sigmoid(self.feature_gate(state_batch))
        gated = state_batch * mask
        features = self.encoder(gated)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
