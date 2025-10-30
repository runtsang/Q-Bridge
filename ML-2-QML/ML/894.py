"""Enhanced classical regression model with configurable architecture and noise-aware data generation.

The module mirrors the original QuantumRegression seed but expands the model
to support arbitrary hidden layer sizes, dropout, and batch‑normalisation.
The data generator now accepts a noise level parameter to simulate measurement
uncertainty, making it easier to benchmark against the quantum counterpart.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic regression targets from a superposition‑like function.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise applied to the targets.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(scale=noise_std, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset yielding feature vectors and regression targets.
    """
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
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
    Fully‑connected regression network with optional dropout and batch‑norm.

    Parameters
    ----------
    num_features : int
        Size of the input feature vector.
    hidden_sizes : list[int], optional
        Sizes of hidden layers. Defaults to [32, 16].
    dropout : float, optional
        Dropout probability applied after each hidden layer. Set to 0 to disable.
    batch_norm : bool, optional
        Whether to insert a BatchNorm1d after each hidden layer.
    """
    def __init__(
        self,
        num_features: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [32, 16]
        layers = []
        prev = num_features
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
