"""Enhanced classical regression model with preprocessing and regularization.

This module builds on the seed implementation by adding:
- Feature scaling via StandardScaler.
- Dropout and BatchNorm layers for better generalization.
- Configurable hidden layer sizes.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate data similar to the quantum seed but with added Gaussian noise."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    noise = np.random.normal(scale=0.05, size=y.shape).astype(np.float32)
    return x, (y + noise).astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns scaled features and targets."""
    def __init__(self, samples: int, num_features: int):
        raw_features, raw_labels = generate_superposition_data(num_features, samples)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(raw_features)
        self.labels = raw_labels

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Feed‑forward network with optional dropout and batch‑norm."""
    def __init__(self, num_features: int, hidden_sizes: tuple[int,...] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
