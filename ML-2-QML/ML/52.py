"""Classical regression module with RBF feature mapping and optional quantum branch."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with a superposition‑like pattern.
    The labels are a sinusoidal function of the sum of the features plus a small cosine
    component. Gaussian noise can be added by the caller if desired.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset wrapping the synthetic superposition data. Each sample is a dictionary
    containing the raw features and the target label.
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


class RBFMapper(nn.Module):
    """
    Radial‑basis‑function feature mapper that expands the input feature space
    into a higher‑dimensional space. The mapping is defined by a set of random
    centers and a fixed gamma parameter.
    """
    def __init__(self, in_features: int, out_features: int, gamma: float = 1.0):
        super().__init__()
        self.register_buffer("centers", torch.randn(out_features, in_features))
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)  # (batch, out_features, in_features)
        dist2 = (diff ** 2).sum(-1)  # (batch, out_features)
        return torch.exp(-self.gamma * dist2)


class QModel(nn.Module):
    """
    Classical regression model with an optional RBF feature mapper and a standard
    feed‑forward network. The model can be configured to use a quantum branch
    by passing a quantum module instance to the forward method.
    """
    def __init__(self, num_features: int, rbf_dim: int = 64, rbf_gamma: float = 0.5):
        super().__init__()
        self.mapper = RBFMapper(num_features, rbf_dim, gamma=rbf_gamma)
        self.net = nn.Sequential(
            nn.Linear(rbf_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # state_batch: (batch, num_features)
        features = self.mapper(state_batch)
        return self.net(features).squeeze(-1)
