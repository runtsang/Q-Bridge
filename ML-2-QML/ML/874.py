"""Enhanced classical regression model with residual connections and data augmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data mimicking a superposition‑like relationship.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Target vector of shape (samples,).
    """
    rng = np.random.default_rng()
    X = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch Dataset wrapping the synthetic regression data.
    """

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RegressionModel(nn.Module):
    """
    A lightweight residual neural network for regression.
    """

    def __init__(self, num_features: int, hidden_dim: int = 64):
        super().__init__()
        self.input_layer = nn.Linear(num_features, hidden_dim)
        self.res_block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = self.input_layer(state_batch)
        res = self.res_block(h)
        h = h + res  # residual connection
        h = self.dropout(h)
        return self.output_layer(h).squeeze(-1)


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
