"""Hybrid classical regression model inspired by quantum feature maps and EstimatorQNN."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of superposition states with sinusoidal labels."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset returning states and target values."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridQuantumRegression(nn.Module):
    """Classical neural network that mimics a quantum feature map."""
    def __init__(self, num_features: int, hidden_dim: int = 32):
        super().__init__()
        # Quantum‑inspired feature transformation: apply smooth activations
        self.feature_map = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.SiLU(),  # smooth activation resembling quantum amplitude
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Final regression head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(state_batch)
        return self.head(x).squeeze(-1)

__all__ = ["HybridQuantumRegression", "RegressionDataset", "generate_superposition_data"]
