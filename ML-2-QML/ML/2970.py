"""Hybrid QCNN regression – classical implementation."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

__all__ = ["HybridQCNNRegression", "RegressionDataset", "generate_superposition_data"]


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset.
    The input is a real‑valued feature vector; the target is a smooth
    non‑linear function of the sum of angles, mimicking a quantum superposition.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset for the hybrid QCNN regression task."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridQCNNRegression(nn.Module):
    """
    Classical QCNN‑style network for supervised regression.
    The architecture mirrors the quantum QCNN layers but uses dense
    layers instead of parameterised gates.
    """
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return self.head(x).squeeze(-1)


def HybridQCNNRegressionFactory(input_dim: int = 8) -> HybridQCNNRegression:
    """Return a pre‑configured instance of the hybrid regression model."""
    return HybridQCNNRegression(input_dim)
