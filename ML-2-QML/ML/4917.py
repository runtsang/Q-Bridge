"""QuantumRegression: Classical regression baseline for superposition data.

The module is intentionally lightweight, mirroring the
``EstimatorQNN`` architecture but operating on the same
superposition data generator used by the quantum branch.  The
model can be expanded with additional feature extractors
(e.g. convolutional layers) without changing the API.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce synthetic data where each sample is a superposition
    of a basis state and its complement.  The target is a
    trigonometric function of the underlying angles.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset wrapper that yields feature vectors and scalar labels.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "data": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class EstimatorNN(nn.Module):
    """
    Tiny feed‑forward network inspired by the EstimatorQNN example.
    It accepts a 2‑dimensional input; for higher dimensional data
    a projection layer can be added externally.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


class QuantumRegression(nn.Module):
    """
    Classical regression model that uses the EstimatorNN as a feature
    extractor followed by a trivial linear head.  The model is
    intentionally minimal to keep the focus on the quantum variant.
    """
    def __init__(self, input_dim: int = 2) -> None:
        super().__init__()
        self.feature_extractor = EstimatorNN()
        # If input_dim differs from 2, a projection layer can be added
        # before the EstimatorNN.
        self.head = nn.Linear(1, 1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        ``data`` must be of shape (batch, 2).  For higher‑dimensional
        data, reshape or project the input before calling this method.
        """
        feat = self.feature_extractor(data)
        return self.head(feat).squeeze(-1)


__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
