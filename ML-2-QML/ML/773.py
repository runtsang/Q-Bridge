"""Classical regression dataset and neural network with residual connections and feature scaling.

The module mirrors the quantum example but adds a preprocessing pipeline
and a deeper residual network.  The dataset generator produces
noise‑augmented samples, and the Dataset class exposes a
`get_dataloader` helper for quick experimentation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression targets from a trigonometric superposition
    of the input features.  A small Gaussian noise term is added to make the
    problem non‑trivial for the network.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    noise = 0.05 * np.random.randn(samples).astype(np.float32)
    return x, (y + noise).astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns a single sample as a dict."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

    def get_dataloader(self, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        """Convenience wrapper to get a DataLoader."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

class ResidualBlock(nn.Module):
    """A simple residual block with two linear layers."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class QModel(nn.Module):
    """
    A deeper feed‑forward network with residual connections.

    The architecture is intentionally more expressive than the seed
    model to demonstrate how classical networks can be scaled.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)
