"""Classical regression model with residual blocks."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = np.sum(x, axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset for superposition regression."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ResidualBlock(nn.Module):
    """Simple residual block."""
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear2(self.relu(self.linear1(x)))

class QModel(nn.Module):
    """Deep residual regression network."""
    def __init__(self, num_features: int, hidden_dim: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(num_features, hidden_dim), nn.ReLU()]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
