"""Enhanced classical regression model with residual blocks and dropout."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data using superposition-inspired features."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and target values."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """Simple residual block with linear layers and ReLU."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = self.norm(out)
        return nn.functional.relu(out + x)


class QModel(nn.Module):
    """Hybrid MLP with residual blocks for regression."""
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = [nn.Linear(num_features, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden_dim))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
