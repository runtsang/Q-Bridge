"""Quantum regression model using Pennylane variational circuit with learnable depth."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as pnp


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum state vectors and targets.

    Each state is a superposition of |0...0> and |1...1> with random angles.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(complex), labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping the quantum state vectors and targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegressionModel(nn.Module):
    """Variational quantum circuit with a learnable depth and a classical head."""

    def __init__(self, num_wires: int, depth: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.depth = depth
        # Variational parameters: depth x num_wires x 2 (Ry, Rz)
        self.params = nn.Parameter(torch.randn(depth, num_wires, 2))
        self.head = nn.Linear(num_wires, 1)

        # Pennylane device (default.qubit) with automatic differentiation
        self.dev = qml.device("default.qubit", wires=num_wires, shots=None)

        # QNode with batch support
        @qml.qnode(self.dev, interface="torch", diff_method="backprop", batch=True)
        def circuit(state: torch.Tensor, params: torch.Tensor):
            qml.QubitStateVector(state, wires=range(num_wires))
            for d in range(self.depth):
                for w in range(num_wires):
                    qml.RY(params[d, w, 0], wires=w)
                    qml.RZ(params[d, w, 1], wires=w)
                for w in range(num_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
                qml.CNOT(wires=[num_wires - 1, 0])
            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        self.circuit = circuit

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch: (batch, 2**num_wires)
        features = self.circuit(state_batch, self.params)
        return self.head(features).squeeze(-1)


__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
