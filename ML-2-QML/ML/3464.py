"""Hybrid classical regression model that mirrors the quantum architecture."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data by summing sinusoidal functions of
    input features.  The output is a scalar per sample.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset returning raw feature vectors and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Classical regression model that emulates the QFCModel architecture:
    linear feature extractor → 4‑dimensional bottleneck → regression head.
    """
    def __init__(self, num_features: int):
        super().__init__()
        # Feature extractor: two hidden layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        # Bottleneck to 4 features (mirrors QFCModel's output size)
        self.fc = nn.Sequential(
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state_batch)
        out = self.fc(features)
        return out.squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
