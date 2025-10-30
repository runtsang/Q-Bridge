"""Enhanced classical regression model with deeper architecture and residual connections.

This module mirrors the original QuantumRegression seed but extends the neural
network with batch normalization, GELU activations, dropout, and a residual
skip connection to improve gradient flow and generalisation.  The dataset
generator now supports optional Gaussian noise and reproducibility via a
seed argument.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int,
                                noise_scale: float = 0.0,
                                seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic superposition data.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    samples : int
        Number of samples to generate.
    noise_scale : float, default 0.0
        Standard deviation of Gaussian noise added to the labels.
    seed : int | None, default None
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Features and corresponding labels.
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_scale > 0.0:
        y += noise_scale * np.random.randn(samples)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset wrapper for the superposition data.
    """
    def __init__(self, samples: int, num_features: int,
                 noise_scale: float = 0.0, seed: int | None = None):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_scale=noise_scale, seed=seed
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegressionModel(nn.Module):
    """
    Deep feed‑forward regression network with residual connections and
    regularisation.  The architecture is designed to match the dimensionality
    of the quantum feature space while providing improved expressivity.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        # Residual connection from input to the last hidden layer
        self.residual = nn.Linear(num_features, 64)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.net[:-1](state_batch)
        residual = self.residual(state_batch)
        x = x + residual
        output = self.net[-1](x)
        return output.squeeze(-1)


__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
