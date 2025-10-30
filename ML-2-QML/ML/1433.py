"""Classical regression model with 400‑sample dataset and extended MLP."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data: superposition‑based inputs and sinusoidal targets."""
    # Uniformly sample feature vectors
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    # Compute angles as sum of features
    angles = x.sum(axis=1)
    # Target: sin(angles) + 0.1 * cos(2 * angles)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset returning feature tensors and scalar targets."""
    def __init__(self, samples: int = 400, num_features: int = 10):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Extended MLP for regression."""
    def __init__(self, num_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
