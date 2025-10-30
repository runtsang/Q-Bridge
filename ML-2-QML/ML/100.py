"""Hybrid regression dataset and classical model with feature extractor."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data where the target depends on the sum of features.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.

    Returns
    -------
    X : np.ndarray of shape (samples, num_features)
        Input features.
    y : np.ndarray of shape (samples,)
        Regression targets.
    """
    rng = np.random.default_rng()
    X = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset for the synthetic regression problem.

    The dataset returns a dictionary with keys ``states`` and ``target`` so
    it can be consumed by both the classical and quantum pipelines.
    """

    def __init__(self, samples: int, num_features: int):
        super().__init__()
        self.x, self.y = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.x)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.x[index], dtype=torch.float32),
            "target": torch.tensor(self.y[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Deep feed‑forward network used as a feature extractor before the quantum head.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_units : int, default 64
        Number of hidden units in each hidden layer.
    depth : int, default 4
        Number of hidden layers.
    """

    def __init__(self, input_dim: int, hidden_units: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_units), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.ReLU()])
        layers.append(nn.Linear(hidden_units, 1))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
