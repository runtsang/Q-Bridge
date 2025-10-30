"""Hybrid regression model combining classical feed-forward and quantum-inspired layers.

This module defines a dataset that produces both classical feature vectors and
corresponding quantum states, and a purely classical neural network that
mimics the structure of the quantum encoder by using a random linear layer
followed by trainable rotation layers and a measurement head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def generate_hybrid_data(num_features: int, num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.

    Parameters
    ----------
    num_features : int
        Dimensionality of the classical feature vector.
    num_wires : int
        Number of qubits used for the quantum state.
    samples : int
        Number of samples.

    Returns
    -------
    features : np.ndarray
        Classical feature matrix of shape (samples, num_features).
    states : np.ndarray
        Complex state vectors of shape (samples, 2**num_wires).
    labels : np.ndarray
        Regression targets of shape (samples,).
    """
    # Classical features uniformly sampled in [-1, 1]
    features = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)

    # Map features to angles for quantum state generation
    thetas = np.arccos(features[:, :1].sum(axis=1))  # simple mapping
    phis = np.arctan2(features[:, 1:2].sum(axis=1), 1.0)

    # Build superposition states |ψ> = cosθ|0...0> + e^{iφ} sinθ|1...1>
    dim = 2 ** num_wires
    states = np.zeros((samples, dim), dtype=complex)
    for i in range(samples):
        states[i, 0] = np.cos(thetas[i])
        states[i, -1] = np.exp(1j * phis[i]) * np.sin(thetas[i])

    # Target is a smooth function of the angles
    labels = np.sin(2 * thetas) * np.cos(phis)

    return features, states.astype(np.complex64), labels.astype(np.float32)

class HybridRegressionDataset(Dataset):
    """Dataset returning classical features, quantum states and labels."""
    def __init__(self, samples: int, num_features: int, num_wires: int):
        self.features, self.states, self.labels = generate_hybrid_data(num_features, num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridQuantumRegression(nn.Module):
    """Purely classical neural network that emulates the quantum encoder."""
    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        # Random linear layer simulating random unitary
        self.random_layer = nn.Linear(num_features, 2 ** num_wires, bias=False)
        nn.init.normal_(self.random_layer.weight, mean=0.0, std=0.1)
        # Trainable rotation layers
        self.rx = nn.Linear(2 ** num_wires, 2 ** num_wires, bias=False)
        self.ry = nn.Linear(2 ** num_wires, 2 ** num_wires, bias=False)
        nn.init.xavier_uniform_(self.rx.weight)
        nn.init.xavier_uniform_(self.ry.weight)
        # Measurement head
        self.head = nn.Linear(2 ** num_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_features)
        z = self.random_layer(x)
        z = torch.tanh(self.rx(z))
        z = torch.tanh(self.ry(z))
        return self.head(z).squeeze(-1)

__all__ = ["HybridQuantumRegression", "HybridRegressionDataset", "generate_hybrid_data"]
