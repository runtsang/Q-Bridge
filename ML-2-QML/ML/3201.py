"""Hybrid QCNN regression model purely classical."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic dataset of superposition‑like feature vectors and a
    regression target.  The target is a smooth nonlinear function of the sum
    of the features, mirroring the quantum regression example.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))


class RegressionDataset(Dataset):
    """Dataset wrapping the superposition data for PyTorch."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.targets = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {"states": self.features[index], "target": self.targets[index]}


class QCNNRegressionModel(nn.Module):
    """
    Classical QCNN‑style regression network that emulates the layer structure
    of the quantum QCNN while staying entirely on CPU/GPU.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(num_features, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return self.head(x).squeeze(-1)


def QCNNRegression() -> QCNNRegressionModel:
    """Factory returning a ready‑to‑train QCNN regression model."""
    return QCNNRegressionModel(num_features=8)


__all__ = ["QCNNRegression", "QCNNRegressionModel", "RegressionDataset", "generate_superposition_data"]
