"""QuantumRegression__gen082.py – classical regression with multi‑head and dropout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from functools import partial

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]


def generate_superposition_data(
    num_features: int,
    samples: int,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics the quantum‑like superposition
    state. The output is a pair (x, y) where y = sin(sum(x)) + 0.1*cos(2*sum(x))
    with optional Gaussian noise.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(scale=noise_std, size=y.shape).astype(np.float32)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset for classical regression. Each item is a dict containing:

    - ``states``: feature vector as a torch tensor (float32)
    - ``target``: regression target as a torch tensor (float32)
    """

    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(
            num_features=num_features,
            samples=samples,
            noise_std=noise_std,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """
    Multi‑head, dropout‑regularised feed‑forward network.

    The network consists of two parallel heads:

    1. *regression_head* – outputs a scalar prediction.
    2. *auxiliary_head* – outputs a probability that the target lies in the
       upper quartile of the training distribution.  This auxiliary task
       regularises the feature extractor and improves generalisation.

    Dropout is applied after each hidden layer and before each head.
    """

    def __init__(self, num_features: int, dropout: float = 0.1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(32, 1)
        self.auxiliary_head = nn.Linear(32, 1)

    def forward(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state_batch)
        return (
            self.regression_head(features).squeeze(-1),
            torch.sigmoid(self.auxiliary_head(features)).squeeze(-1),
        )
