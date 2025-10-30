"""Classical regression dataset and model with adaptive MLP head and early stopping support."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int, noise_level: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics a superposition‑like pattern.
    The target is sin(theta) + noise, where theta is the sum of features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    theta = x.sum(axis=1)
    y = np.sin(theta) + noise_level * np.random.randn(samples)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumRegression__gen205(nn.Module):
    """
    Adaptive MLP head with early‑stopping support.
    The architecture is: Linear → ReLU → Linear → ReLU → Linear.
    The number of hidden units is determined by num_features.
    """
    def __init__(self, num_features: int, hidden_scale: float = 1.5):
        super().__init__()
        hidden1 = int(num_features * hidden_scale)
        hidden2 = int(hidden1 * hidden_scale)
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

__all__ = ["QuantumRegression__gen205", "RegressionDataset", "generate_superposition_data"]
