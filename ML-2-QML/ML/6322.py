"""Hybrid classical regression model with convolutional feature extraction and fully connected head.

This module extends the original regression seed by adding a lightweight 2‑D CNN
to capture spatial structure in the 16‑dimensional feature vectors, followed by
a fully‑connected head that outputs a scalar target.  The scaling strategy is a
combination of classical batch‑norm and ReLU activations with a quantum‑style
linear head, providing a clear contrast to the pure fully‑connected baseline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int = 16, samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data: a linear combination of sinusoids
    applied to the sum of the feature vector components.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset that returns 4×4 images derived from the original 16‑dimensional
    feature vectors, along with the corresponding regression target.
    """
    def __init__(self, samples: int = 1000, num_features: int = 16):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        image = self.features[index].reshape(1, 4, 4)
        return {
            "states": torch.tensor(image, dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """
    Hybrid classical regression model.
    The convolutional backbone extracts spatial features from the 4×4 images,
    which are then passed through a fully‑connected head to produce a scalar
    regression output.
    """
    def __init__(self, num_features: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened).squeeze(-1)
        return self.norm(out.unsqueeze(-1)).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
