"""Advanced classical regression model with polynomial feature engineering.

The module keeps the original data generation and dataset classes but replaces
the shallow feed‑forward network with a polynomial feature extractor followed
by a deeper MLP.  This allows the model to capture higher‑order interactions
without changing the external interface.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset.

    The data mimics a simple quantum state superposition: the target is a
    nonlinear function of the sum of the features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns a state vector and its target value."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class AdvancedRegressionModel(nn.Module):
    """MLP that ingests polynomial features of the input state."""

    def __init__(self, num_features: int):
        super().__init__()
        # Number of unique polynomial terms (x_i * x_j for i <= j)
        poly_features = num_features * (num_features + 1) // 2
        input_dim = num_features + poly_features

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _poly_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return concatenated raw + pairwise product features."""
        # Raw features
        raw = x
        # Pairwise products
        prod = (x.unsqueeze(2) * x.unsqueeze(1)).reshape(x.shape[0], -1)
        # Keep only upper triangular (including diagonal)
        idx = torch.triu_indices(x.shape[1], x.shape[1], offset=0)
        prod = prod[:, idx[0] * x.shape[1] + idx[1]]
        return torch.cat([raw, prod], dim=1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self._poly_features(state_batch)
        return self.net(features).squeeze(-1)


__all__ = ["AdvancedRegressionModel", "RegressionDataset", "generate_superposition_data"]
