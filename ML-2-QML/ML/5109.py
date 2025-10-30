"""Hybrid regression model with classical backbone and shared data generation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data that can be fed to either the classical or quantum model.
    The target is sin(θ) + 0.1 cos(2θ) where θ is the sum of the feature vector.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary containing the raw state and target.
    The same data can be used for the quantum encoder or the classical net.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class ClassicalBackbone(nn.Module):
    """
    Dense network that captures linear and low‑order polynomial patterns.
    """
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze(-1)

class QuantumEncoder(nn.Module):
    """
    Variational encoder that maps the classical input into a quantum state
    and produces a feature vector from Pauli‑Z measurements.
    """
    def __init__(self, num_wires: int, n_layers: int = 3, n_params_per_layer: int = 2):
        super().__init__()
        self.n_wires = num_wires
        self.n_layers = n_layers
        self.n_params_per_layer = n_params_per_layer

        # Parameterised rotation layers
        self.rxs = nn.ParameterList([
            nn.Parameter(torch.randn(num_wires)) for _ in range(n_layers)
        ])
        self.rys = nn.ParameterList([
            nn.Parameter(torch.randn(num_wires)) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulated quantum circuit: apply Ry and Rx per wire, then
        perform a measurement that returns the expected Pauli‑Z.
        """
        batch_size = x.shape[0]
        # encode the input as angles
        angles = x  # treat each feature as an angle
        # compute expectation of Pauli‑Z for each wire
        probs = 1 - 2 * torch.sin(2 * angles) ** 2
        # 1st layer
        for i in range(self.n_layers):
            probs = probs * torch.cos(self.rxs[i])  # simple mixing
            probs = probs * torch.cos(self.rys[i])
        return probs.mean(dim=1)

class QuantumRegressionEnhanced(nn.Module):
    """
    Hybrid model that concatenates classical and quantum features
    and trains a final linear head.
    """
    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.classical = ClassicalBackbone(num_features)
        self.quantum = QuantumEncoder(num_wires)
        self.head = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        klass = self.classical(x)
        qfeat = self.quantum(x)
        combined = torch.stack([klass, qfeat], dim=-1)
        return self.head(combined).squeeze(-1)

__all__ = ["QuantumRegressionEnhanced", "RegressionDataset", "generate_superposition_data"]
