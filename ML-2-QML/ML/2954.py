"""Classical regression engine that mirrors the quantum architecture.

The module defines a data generator, a dataset wrapper, and a neural network
EstimatorQNN__gen221 that matches the shape of the quantum model.  The
encoder mimics the random quantum layer with linear transformations and
ReLU activations, enabling direct performance comparison without quantum
hardware.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data that emulates the quantum state
    superposition pattern: y = sin(sum(x)) + 0.1 * cos(2 * sum(x)).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns a feature vector and its regression target."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class EstimatorQNN__gen221(nn.Module):
    """
    Classical feed‑forward network that mimics the quantum encoder.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    hidden_size : int, default 64
        Size of the hidden layers.
    """

    def __init__(self, num_features: int = 2, hidden_size: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.encoder(inputs)
        return self.head(features).squeeze(-1)


__all__ = ["EstimatorQNN__gen221", "RegressionDataset", "generate_superposition_data"]
