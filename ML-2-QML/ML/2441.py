"""Hybrid classification/regression model combining CNN and regression head, inspired by QuantumNAT and QuantumRegression."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class QuantumNATRegressionModel(nn.Module):
    """
    A hybrid model that merges the CNN backbone of QuantumNAT with a regression head
    from QuantumRegression. The model can be used for image classification (output 4)
    or scalar regression (output 1) depending on the `task` flag.
    """
    def __init__(self, task: str = "classification") -> None:
        super().__init__()
        self.task = task
        # CNN backbone identical to QFCModel
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Fully connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4 if task == "classification" else 1),
        )
        self.norm = nn.BatchNorm1d(4 if task == "classification" else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic superposition data for regression, mirroring QuantumRegression.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns superposition states and targets for regression.
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

__all__ = ["QuantumNATRegressionModel", "RegressionDataset", "generate_superposition_data"]
