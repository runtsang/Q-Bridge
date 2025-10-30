"""Extended classical regression dataset and model using a deeper MLP with dropout and batch norm."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data mirroring quantum superposition but with added Gaussian noise.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y_clean = np.sin(angles) + 0.1 * np.cos(2 * angles)
    noise = np.random.normal(0, 0.05, size=y_clean.shape).astype(np.float32)
    return x, (y_clean + noise).astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset returning feature vectors and noisy target values.
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

class QModel(nn.Module):
    """
    Deep MLP with residual connections, batch‑norm and dropout for robust regression.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Residual shortcut
        self.residual = nn.Linear(num_features, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.net(state_batch.to(torch.float32)).squeeze(-1)
        return out + self.residual(state_batch)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
