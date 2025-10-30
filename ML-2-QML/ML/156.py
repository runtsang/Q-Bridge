"""Classical regression dataset and model mirroring the quantum example with enhanced architecture."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data where the target is a smooth non‑linear function of the sum of features,
    with optional Gaussian noise.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature space.
    samples : int
        Number of samples to generate.
    noise_std : float, default 0.05
        Standard deviation of Gaussian noise added to the target.

    Returns
    -------
    features : np.ndarray, shape (samples, num_features)
        Uniformly sampled features in [-1, 1].
    labels : np.ndarray, shape (samples,)
        Non‑linear target values with noise.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += np.random.normal(scale=noise_std, size=y.shape)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary with feature tensor and scalar target.
    """
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.05):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """
    A flexible feed‑forward network with optional dropout and batch‑norm.
    """
    def __init__(self, num_features: int, hidden_sizes: list[int] | tuple[int,...] = (64, 32), dropout_prob: float = 0.2):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
