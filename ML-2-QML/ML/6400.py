"""Hybrid sampler and regression neural network combining classical and quantum-inspired designs.

This module merges the classical sampler network from SamplerQNN.py with the regression
architecture from QuantumRegression.py.  It exposes a single `HybridSamplerRegressor`
class that can operate in either sampling or regression mode, controlled by the
`mode` argument to `forward`.  The sampler part mirrors the original 2‑to‑2
softmax network; the regression part follows the 3‑layer regression network
used in the seed.  The design is intentionally modular so the same class can
be swapped into pipelines that previously used either seed standalone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for regression, mirroring the quantum seed.

    The data distribution is the same as in the original QuantumRegression seed
    but produced as a NumPy array so it can be used by both classical and
    quantum pipelines.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic regression data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridSamplerRegressor(nn.Module):
    """
    Dual‑purpose network.

    Parameters
    ----------
    num_features : int
        Number of input features for the regression head.
    mode : str, optional
        ``'sample'`` or ``'regress'``.  The default is ``'sample'``.
    """

    def __init__(self, num_features: int = 2, mode: str = "sample") -> None:
        super().__init__()
        self.mode = mode

        # Sampler head (original 2‑to‑2 softmax)
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

        # Regression head (original 3‑layer network)
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def set_mode(self, mode: str) -> None:
        """Switch between sampling and regression."""
        assert mode in {"sample", "regress"}, "mode must be'sample' or'regress'"
        self.mode = mode

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.mode == "sample":
            return F.softmax(self.sampler(inputs), dim=-1)
        else:
            return self.regressor(inputs).squeeze(-1)


__all__ = ["HybridSamplerRegressor", "RegressionDataset", "generate_superposition_data"]
