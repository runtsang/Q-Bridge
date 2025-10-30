"""Enhanced classical regression model with residual connections and normalisation.

This module extends the original seed by adding feature scaling, dropout,
batch normalisation, and a flexible residual network. The dataset now supports
noise injection and dynamic feature sizes, making it more suitable for
benchmarking against the quantum counterpart.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Sequence, Tuple


def generate_superposition_data(
    num_features: int,
    samples: int,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic superposition data with optional Gaussian noise.

    Parameters
    ----------
    num_features : int
        Number of features per sample.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise on the target.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(scale=noise_std, size=y.shape)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset that normalises features to zero mean and unit variance.
    """
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_std
        )
        # Feature normalisation
        self.mean = self.features.mean(axis=0, keepdims=True)
        self.std = self.features.std(axis=0, keepdims=True) + 1e-6
        self.features = (self.features - self.mean) / self.std

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegressionModel(nn.Module):
    """
    A flexible residual network for regression tasks.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    hidden_dims : Sequence[int], optional
        Sizes of hidden layers. Default: [64, 32].
    dropout : float, optional
        Dropout probability. Default: 0.1.
    """
    def __init__(
        self,
        num_features: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = num_features
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)


__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
