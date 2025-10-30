"""Enhanced classical regression pipeline with residual blocks and normalisation."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int,
                                noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where the target is a non‑linear
    function of the input features.  Adds optional Gaussian noise.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    samples : int
        Number of examples to generate.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the target.

    Returns
    -------
    x : np.ndarray
        Shape (samples, num_features) feature matrix.
    y : np.ndarray
        Shape (samples,) target vector.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(scale=noise_std, size=y.shape)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapper for the synthetic superposition data."""

    def __init__(self, samples: int, num_features: int,
                 noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_std
        )
        # Normalise features to zero mean and unit variance
        self.features = (self.features - self.features.mean(axis=0)) / (
            self.features.std(axis=0) + 1e-6
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RegressionModel(nn.Module):
    """
    Feed‑forward network with residual blocks and dropout.

    The architecture is intentionally deeper than the seed while remaining
    lightweight enough for quick experimentation.
    """

    def __init__(self, num_features: int, hidden_dim: int = 64,
                 dropout_prob: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        # Residual shortcut
        self.residual = nn.Linear(num_features, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.net(state_batch)
        res = self.residual(state_batch)
        out = out + res
        return self.head(out).squeeze(-1)


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
