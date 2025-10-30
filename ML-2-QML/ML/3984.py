"""EstimatorQNN__gen304 – Classical regression module.

Combines the simple feed‑forward network from the original EstimatorQNN
with the synthetic superposition data and deeper architecture from
QuantumRegression.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data where the target depends on the sum of the
    input features in a sinusoidal way.  Mirrors the function from
    QuantumRegression.py but stays entirely classical.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that yields (feature, target) pairs."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return (
            torch.tensor(self.features[index], dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


class EstimatorQNN(nn.Module):
    """
    Classical fully‑connected regressor.
    Layer sizes: 2 → 64 → 32 → 16 → 1 with ReLU activations and dropout.
    """

    def __init__(self, input_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze(-1)


def get_dataloader(
    samples: int = 1024,
    batch_size: int = 32,
    num_features: int = 2,
) -> DataLoader:
    """
    Helper that returns a DataLoader for training or evaluation.
    """
    dataset = RegressionDataset(samples, num_features)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


__all__ = [
    "EstimatorQNN",
    "RegressionDataset",
    "generate_superposition_data",
    "get_dataloader",
]
