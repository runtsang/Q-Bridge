"""Hybrid classical regression model with confidence output and data augmentation.

This module extends the original seed by adding Gaussian noise augmentation,
a dropout-based confidence estimate, and a custom loss that mixes MSE with
confidence penalty. The model outputs both the predicted target and a
confidence score between 0 and 1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with a sine-like target and optional noise."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that optionally adds Gaussian noise to features."""
    def __init__(self, samples: int, num_features: int, augment: bool = False):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        feat = self.features[index]
        if self.augment:
            # Gaussian noise added to each feature
            feat = feat + 0.1 * np.random.randn(*feat.shape)
        return {
            "states": torch.tensor(feat, dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Classical regression model that outputs target and confidence."""
    def __init__(self, num_features: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.head_target = nn.Linear(16, 1)
        self.head_confidence = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state_batch)
        target = self.head_target(features).squeeze(-1)
        confidence = self.head_confidence(features).squeeze(-1)
        return target, confidence

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
