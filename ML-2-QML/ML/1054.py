"""Hybrid classical regression module with a learnable feature extractor and a linear head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data where the target is a non‑linear function of the sum
    of features.  The data are sampled from a uniform distribution in
    [-1, 1] and the labels are defined as
    ``y = sin(sum(x)) + 0.1 * cos(2 * sum(x))``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = np.sum(x, axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class FeatureExtractor(nn.Module):
    """
    A shallow feed‑forward network that learns a compact representation
    of the input features before they are passed to a regression head.
    """
    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ClassicalRegressionModel(nn.Module):
    """
    Classical regression model that uses a feature extractor followed by a
    linear head.  The design mirrors the quantum version but remains fully
    classical, enabling direct comparisons.
    """
    def __init__(self, num_features: int, hidden: int = 64):
        super().__init__()
        self.extractor = FeatureExtractor(num_features, hidden)
        self.head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extractor(x)
        return self.head(feat).squeeze(-1)

__all__ = ["FeatureExtractor", "ClassicalRegressionModel", "RegressionDataset", "generate_superposition_data"]
