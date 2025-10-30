"""Quantum regression model using PennyLane with a fully parameterised variational circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states of the form cos(θ)|0…0⟩ + e^{iϕ} sin(θ)|1…1⟩."""
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
    """Dataset wrapping the quantum regression data."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RegressionModel(nn.Module):
    """Quantum‑classical hybrid model with a trainable variational circuit."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        # Total parameters: 3 rotations per qubit + (num_wires-1) entangling params
        self.num_params = num_wires * 3 + (num_wires - 1)
        self.params = nn.Parameter(torch.randn(self.num_params))
        self.head = nn.Linear(num_wires, 1)

    def _quantum_circuit(self, state: torch.Tensor, params: torch.Tensor):
        """PennyLane QNode that prepares the state and applies a variational layer."""
        dev = qml.device("default.qubit", wires=self.num_wires, shots=0, batch_size=state.shape[0])

        @qml.qnode(dev, interface="torch")
        def circuit(state_vec, param_vec):
            # State preparation
            qml.StatePrep(state_vec, wires=range(self.num_wires))
            idx = 0
            # Parameterised single‑qubit rotations
            for i in range(self.num_wires):
                qml.RY(param_vec[idx], wires=i)
                idx += 1
                qml.RX(param_vec[idx], wires=i)
                idx += 1
                qml.RZ(param_vec[idx], wires=i)
                idx += 1
            # Entangling layer
            for i in range(self.num_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Expectation values of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

        return circuit(state, params)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        features = self._quantum_circuit(state_batch, self.params)
        return self.head(features).squeeze(-1)


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
