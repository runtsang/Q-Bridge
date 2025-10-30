"""Classical regression dataset and model mirroring the quantum example with enhanced architecture."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data mimicking a quantum superposition."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapper for the synthetic regression data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QuantumRegression__gen305(nn.Module):
    """Enhanced classical regression model with residual skip and batch‑norm."""

    def __init__(self, num_features: int):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(num_features)
        self.network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.residual = nn.Linear(num_features, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(state_batch)
        out = self.network(x)
        res = self.residual(state_batch)
        combined = out + res
        return self.output(combined).squeeze(-1)


__all__ = ["QuantumRegression__gen305", "RegressionDataset", "generate_superposition_data"]
