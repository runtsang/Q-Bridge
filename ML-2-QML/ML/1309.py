"""Enhanced classical regression model with residual connections and feature scaling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data with a non‑linear target.

    The target is a combination of sine and cosine terms of the summed
    input features, with Gaussian noise added.  This mirrors the
    quantum‑style superposition data while remaining purely classical.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles) + 0.05 * np.random.randn(samples)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Torch dataset yielding feature vectors and scalar targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features).astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """Simple residual block with two linear layers."""
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
    """Deep residual regression model."""
    def __init__(self, num_features: int, hidden_dim: int = 64, num_blocks: int = 3):
        super().__init__()
        layers = [nn.Linear(num_features, hidden_dim), nn.ReLU()]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
