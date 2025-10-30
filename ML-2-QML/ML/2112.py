"""Hybrid classical regression model based on the original seed.

The module adds a trainable feature extractor followed by a linear head.
It keeps the dataset generator unchanged for direct comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Dataset utilities – identical to the original seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return feature matrix and target vector.

    The function is deliberately kept the same as in the seed to allow
    comparison across experiments.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Simple torch Dataset for the synthetic regression data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Hybrid classical regression model
# --------------------------------------------------------------------------- #
class HybridRegressionModel(nn.Module):
    """A classical MLP that learns a feature extractor followed by a linear head."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 32,
        output_dim: int = 1,
        *,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)
        self.to(device)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature extractor and head."""
        feats = self.feature_extractor(state_batch)
        return self.head(feats).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
