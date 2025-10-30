"""Enhanced classical regression model with robust feature processing and regularization.

The module defines a synthetic dataset generator that simulates superposition‑like
features, a PyTorch Dataset wrapper and a two–hidden‑layer neural network with
batch‑normalization and dropout.  The model is ready for end‑to‑end training
within a standard PyTorch workflow.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int, *,
                                noise_std: float = 0.05,
                                seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data mimicking a quantum superposition.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the labels.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    x, y : np.ndarray
        ``x`` has shape ``(samples, num_features)`` and ``y`` has shape
        ``(samples,)``.  The labels are a noisy sinusoidal function of the
        sum of the features.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y_clean = np.sin(angles) + 0.1 * np.cos(2 * angles)
    noise = rng.normal(scale=noise_std, size=angles.shape).astype(np.float32)
    y = y_clean + noise
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """PyTorch Dataset wrapping the synthetic regression data."""

    def __init__(self, samples: int, num_features: int, *,
                 noise_std: float = 0.05, seed: int | None = None):
        self.features, self.labels = generate_superposition_data(
            num_features, samples,
            noise_std=noise_std,
            seed=seed,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegression(nn.Module):
    """Two‑hidden‑layer feed‑forward network with batch‑norm and dropout.

    The architecture is intentionally simple yet expressive enough for
    regression on the synthetic dataset.  Dropout and batch‑norm help
    mitigate over‑fitting and improve generalisation.
    """

    def __init__(self, num_features: int, *,
                 hidden_units: tuple[int, int] = (64, 32),
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_units[0]),
            nn.BatchNorm1d(hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.BatchNorm1d(hidden_units[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units[1], 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
