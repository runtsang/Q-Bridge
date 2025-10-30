"""Hybrid regression dataset and classical model.

The module defines a dataset that exposes both classical feature vectors and
corresponding quantum states, and a lightweight feed‑forward network that
operates solely on the classical side.  The design mirrors the structure of
the original QuantumRegression example while adding a second data channel
for quantum experiments.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce classical features, quantum states and regression targets.

    * Classical features: uniformly sampled in [-1, 1].
    * Quantum states: superposition of |0…0⟩ and |1…1⟩ with random amplitudes.
    * Target: a smooth non‑linear function of the angles used to build the state.
    """
    # classical features
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)

    # quantum states
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1

    # target function
    y = np.sin(2 * thetas) * np.cos(phis)

    return x, states, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset exposing both classical and quantum data channels.

    Parameters
    ----------
    samples : int
        Number of samples to generate.
    num_features : int
        Dimensionality of the classical feature vector.
    num_wires : int
        Number of qubits used to build the quantum state.
    """

    def __init__(self, samples: int, num_features: int, num_wires: int):
        self.features, self.states, self.labels = generate_superposition_data(
            num_features, num_wires, samples
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.labels)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[index], dtype=torch.float32),
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegressionModel(nn.Module):
    """Simple feed‑forward regression network operating on the classical
    feature vector.  It is intentionally lightweight so that it can serve as
    a baseline when experimenting with the quantum counterpart.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
