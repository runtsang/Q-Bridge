"""Hybrid classical regression model combining a quantum‑inspired layer and a neural network.

This module mirrors the original QuantumRegression example but augments the
classical network with a differentiable layer that mimics the behaviour of
the fully‑connected quantum circuit from the second reference pair.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data where the target is a non‑linear function of
    the sum of the input features.  The function mirrors the quantum
    superposition used in the QML seed, providing a fair comparison
    between the classical and quantum models."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class HybridRegressionDataset(Dataset):
    """Dataset that returns the same feature/label pairs as the quantum
    dataset but in a pure NumPy / Torch format."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumInspiredLayer(nn.Module):
    """A differentiable surrogate for the fully‑connected quantum layer.
    It applies a linear transform followed by a tanh non‑linearity and
    averages the result across the feature dimension, mimicking the
    expectation value computed by a quantum circuit."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas shape: (batch, n_features)
        values = self.linear(thetas)
        expectation = torch.tanh(values).mean(dim=1, keepdim=True)
        return expectation

class HybridRegressionModel(nn.Module):
    """Classical regression model that concatenates the output of the
    quantum‑inspired layer with the original input features before
    feeding them to a small MLP."""
    def __init__(self, num_features: int):
        super().__init__()
        self.q_layer = QuantumInspiredLayer(num_features)
        self.mlp = nn.Sequential(
            nn.Linear(num_features + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_feat = self.q_layer(x)
        combined = torch.cat([x, q_feat], dim=1)
        return self.mlp(combined).squeeze(-1)

__all__ = [
    "HybridRegressionDataset",
    "HybridRegressionModel",
    "generate_superposition_data",
]
