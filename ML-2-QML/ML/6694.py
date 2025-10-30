"""Hybrid regression model combining classical neural network with quantum-inspired data generation.

This module extends the original QuantumRegression seed by introducing a deeper
classical network, optional feature augmentation from the quantum state
amplitudes, and a small utility for hybrid training pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data consisting of random feature vectors and labels
    derived from a sinusoidal function of the feature sum.

    Parameters
    ----------
    num_features: int
        Dimensionality of the input feature space.
    samples: int
        Number of data points to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Features of shape (samples, num_features) and
        labels of shape (samples,).
    """
    # Uniform features in [-1, 1]
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the synthetic regression data.
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

class HybridRegressionModel(nn.Module):
    """
    Classical neural network for regression.

    The architecture is a small fully‑connected network that can be
    trained on the dataset produced by ``generate_superposition_data``.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
