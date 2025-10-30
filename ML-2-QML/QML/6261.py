"""Hybrid quantum regression model using PennyLane Gaussian operations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Iterable

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate complex superposition states."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridQuantumRegressionModel(nn.Module):
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.dev = qml.device("default.gaussian", wires=self.num_wires)

        # Define the variational circuit as a TorchLayer
        self.vqc = qml.qnn.TorchLayer(
            self._construct_circuit,
            {
                "squeeze": [torch.randn(self.num_wires)],
                "phase": [torch.randn(self.num_wires)],
                "disp": [torch.randn(self.num_wires)],
                "squeeze2": [torch.randn(self.num_wires)],
                "kerr": [torch.randn(self.num_wires)],
            },
            device=self.dev,
        )
        self.head = nn.Linear(self.num_wires, 1)

    def _construct_circuit(self, x, weights):
        # Encode classical features as displacement on each mode
        for i in range(self.num_wires):
            qml.Displacement(x[i], 0.0, wires=i)
        # Random layer: Gaussian squeezing and phase shift
        squeeze = weights["squeeze"][0]
        phase = weights["phase"][0]
        for i in range(self.num_wires):
            qml.Squeezing(squeeze[i], 0.0, wires=i)
            qml.PhaseShift(phase[i], wires=i)
        # Photonic‑inspired layer: displacement, squeezing, Kerr
        disp = weights["disp"][0]
        squeeze2 = weights["squeeze2"][0]
        kerr = weights["kerr"][0]
        for i in range(self.num_wires):
            qml.Displacement(disp[i], 0.0, wires=i)
            qml.Squeezing(squeeze2[i], 0.0, wires=i)
            qml.Kerr(kerr[i], wires=i)
        # Measurement: expectation of X quadrature
        return [qml.expval(qml.X(i)) for i in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Convert complex state to real feature vector
        features = torch.real(state_batch)[:, :self.num_wires]
        quantum_features = self.vqc(features)
        return self.head(quantum_features).squeeze(-1)

# Alias for compatibility
QModel = HybridQuantumRegressionModel

__all__ = [
    "HybridQuantumRegressionModel",
    "QModel",
    "RegressionDataset",
    "generate_superposition_data",
]
