"""Hybrid classical estimator combining convolutional feature extraction, regression, and sampling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    """Dataset generating superposition states as classical features."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = self._generate_data(num_features, samples)

    @staticmethod
    def _generate_data(num_features: int, samples: int):
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QFCFeatureExtractor(nn.Module):
    """A lightweight CNN followed by a linear projection."""
    def __init__(self, in_channels: int = 1, out_features: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.norm(x)

class ClassicalSampler(nn.Module):
    """Softmax sampler network."""
    def __init__(self, in_features: int = 2, out_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 4),
            nn.Tanh(),
            nn.Linear(4, out_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class EstimatorQNNGen449(nn.Module):
    """Combined estimator that extracts features, regresses, and samples."""
    def __init__(self, num_features: int = 2):
        super().__init__()
        self.feature_extractor = QFCFeatureExtractor()
        self.regressor = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.sampler = ClassicalSampler()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        pred = self.regressor(features)
        probs = self.sampler(features)
        return {"prediction": pred.squeeze(-1), "probabilities": probs}

__all__ = ["EstimatorQNNGen449", "RegressionDataset"]
