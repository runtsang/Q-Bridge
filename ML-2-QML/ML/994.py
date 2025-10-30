"""Enhanced classical regression model with residual connections and data augmentation.

This module extends the original simple feed‑forward network by:
- Adding batch‑normalization and dropout for better regularisation.
- Introducing a residual block to ease optimisation.
- Providing a flexible data‑generator that supports additive Gaussian noise.
- Exposing a convenient `get_dataloaders` helper that splits data into train/val sets.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader


def generate_superposition_data(
    num_features: int,
    samples: int,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data using a superposition‑like function.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise applied to the target.

    Returns
    -------
    x : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Target vector of shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(scale=noise_std, size=y.shape)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset wrapper for the synthetic regression data.

    The dataset returns a dictionary with keys ``states`` and ``target`` to
    stay consistent with the quantum counterpart.
    """

    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_std
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """A simple residual block with two linear layers and a ReLU."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class QModel(nn.Module):
    """
    Classical regression model with residual connections and dropout.
    """

    def __init__(self, num_features: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


def get_dataloaders(
    samples: int,
    num_features: int,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    noise_std: float = 0.0,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Convenience helper that returns training and validation DataLoaders.

    Parameters
    ----------
    samples : int
        Total number of samples.
    num_features : int
        Feature dimensionality.
    batch_size : int
        Batch size for DataLoaders.
    val_ratio : float
        Fraction of data reserved for validation.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    shuffle : bool
        Whether to shuffle the dataset before splitting.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    dataset = RegressionDataset(samples, num_features, noise_std)
    if seed is not None:
        torch.manual_seed(seed)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


__all__ = [
    "QModel",
    "RegressionDataset",
    "generate_superposition_data",
    "get_dataloaders",
]
