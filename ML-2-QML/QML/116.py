"""Advanced quantum regression model using PennyLane.

The QML module retains the same dataset and data‑generation utilities but
introduces a variational circuit with parameterised rotations and a
full‑entanglement layer.  A PennyLane QNode is wrapped in a Torch module,
allowing end‑to‑end training with back‑propagation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset using a quantum superposition.

    The states are of the form |ψ⟩ = cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩.
    The target is a simple trigonometric function of θ and φ.
    """
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
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns a quantum state vector and its target value."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class AdvancedRegressionModel(nn.Module):
    """Variational quantum circuit wrapped as a Torch module."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.dev = qml.device("default.qubit", wires=num_wires)

        # Parameter vector: 2 * num_wires rotation angles
        self.params = nn.Parameter(torch.randn(2 * num_wires))

        # Define the circuit as a PennyLane QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # State preparation
            qml.QubitStateVector(state, wires=range(num_wires))
            # Parameterised rotations
            for i in range(num_wires):
                qml.RX(params[i], wires=i)
                qml.RY(params[i + num_wires], wires=i)
            # Entangle all qubits
            for i in range(num_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation values of Z on each wire
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.circuit = circuit
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch: (batch, 2**num_wires) complex tensor
        # Ensure input is real if using default.qubit
        # The QNode expects a real vector representing the amplitudes
        # Convert complex tensor to real representation if needed
        # Here we assume input is already a valid state vector
        features = self.circuit(state_batch, self.params)
        # features is a list; convert to tensor
        features = torch.stack(features, dim=1)
        return self.head(features).squeeze(-1)


__all__ = ["AdvancedRegressionModel", "RegressionDataset", "generate_superposition_data"]
