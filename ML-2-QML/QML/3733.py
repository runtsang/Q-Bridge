"""Quantum regression module that consumes the angles produced by the classical sampler.

The circuit is a depth‑2 variational ansatz with 4 trainable rotation angles.
A single Pauli‑Z expectation value is used as the quantum feature,
which is then fed into a linear head to produce a scalar prediction.
"""

from __future__ import annotations

import pennylane as qml
import numpy as npy
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class HybridQuantumRegressor(nn.Module):
    """Quantum circuit + linear readout."""
    def __init__(self, device: str = "default.qubit") -> None:
        super().__init__()
        self.dev = qml.device(device, wires=2)
        # Build a PennyLane QNode that accepts two tensors: input angles and weight angles
        self.circuit = qml.QNode(self._circuit, interface="torch", device=self.dev)
        self.head = nn.Linear(1, 1)

    def _circuit(self, input_angles: torch.Tensor, weight_angles: torch.Tensor):
        # Input encoding
        qml.RY(input_angles[0], wires=0)
        qml.RY(input_angles[1], wires=1)
        qml.CNOT(wires=[0, 1])

        # Variational layer 1
        qml.RY(weight_angles[0], wires=0)
        qml.RY(weight_angles[1], wires=1)
        qml.CNOT(wires=[0, 1])

        # Variational layer 2
        qml.RY(weight_angles[2], wires=0)
        qml.RY(weight_angles[3], wires=1)

        # Expectation value as quantum feature
        return qml.expval(qml.PauliZ(0))

    def forward(self, input_angles: torch.Tensor, weight_angles: torch.Tensor) -> torch.Tensor:
        expval = self.circuit(input_angles, weight_angles)
        return self.head(expval.unsqueeze(-1)).squeeze(-1)

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate complex superposition states and labels for quantum regression."""
    omega_0 = npy.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = npy.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * npy.pi * npy.random.rand(samples)
    phis = 2 * npy.pi * npy.random.rand(samples)
    states = npy.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = npy.cos(thetas[i]) * omega_0 + npy.exp(1j * phis[i]) * npy.sin(thetas[i]) * omega_1
    labels = npy.sin(2 * thetas) * npy.cos(phis)
    return states, labels.astype(npy.float32)

class RegressionDataset(Dataset):
    """Dataset yielding quantum states and target values."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

def SamplerQNN() -> HybridQuantumRegressor:
    """Convenience wrapper matching the original anchor."""
    return HybridQuantumRegressor()

__all__ = ["HybridQuantumRegressor", "RegressionDataset",
           "generate_superposition_data", "SamplerQNN"]
