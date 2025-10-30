"""Enhanced classical regression model and dataset."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic *superposition* dataset.
    Features are uniformly sampled in [-1, 1]²ⁿ.
    Labels are a nonlinear combination of their sum.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch Dataset that returns a dictionary containing
    the state vector and its target.
    """
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        if noise_std > 0.0:
            self.features += np.random.normal(scale=noise_std, size=self.features.shape).astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QuantumRegression(nn.Module):
    """
    Classical feed‑forward network with optional dropout and configurable hidden layers.
    """
    def __init__(self, num_features: int, hidden_layers: list[int] | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_layers = hidden_layers or [64, 32]
        layers = []
        in_dim = num_features
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)


__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
