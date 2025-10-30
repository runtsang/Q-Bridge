"""Hybrid QCNN regression – classical implementation.

This module defines a classical QCNN‑style network and a regression
dataset that mirrors the quantum counterpart.  The network is
fully compatible with PyTorch training pipelines and can be used
directly in place of the original QCNN model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class HybridQCNNRegression(nn.Module):
    """
    Classical QCNN-inspired network with a regression head.

    The architecture follows the original QCNN layers but ends with a
    single‑output sigmoid regression.  It is intentionally
    lightweight to allow quick baseline comparisons.
    """
    def __init__(self, num_features: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(num_features, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x)).squeeze(-1)


def QCNN() -> HybridQCNNRegression:
    """
    Factory that returns a ready‑to‑train instance of the classical QCNN.
    """
    return HybridQCNNRegression()


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for regression.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vectors.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Features of shape (samples, num_features) and targets of shape
        (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the synthetic superposition data.
    """
    def __init__(self, samples: int, num_features: int) -> None:
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


__all__ = ["HybridQCNNRegression", "QCNN", "RegressionDataset", "generate_superposition_data"]
