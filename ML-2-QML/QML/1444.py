"""Hybrid quantum‑classical regression model using PennyLane."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0…0〉 + e^{i phi} sin(theta)|1…1〉.
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
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for the quantum superposition data.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QuantumRegression(nn.Module):
    """
    Quantum variational regression model built on PennyLane.
    The encoder maps classical data to rotations; the variational circuit
    contains entangling layers and a trainable linear head.
    """
    def __init__(self, num_wires: int, layers: int = 3, entanglement: str = "full"):
        super().__init__()
        self.num_wires = num_wires
        self.device = qml.device("default.qubit", wires=num_wires, shots=1024)

        # Parameter‑shared encoding
        self.encoder = lambda x, wires: [qml.RX(x[i], wires=wires[i]) for i in range(num_wires)]

        # Variational circuit
        self.var_params = nn.Parameter(torch.randn(layers, num_wires, 3))
        self.entanglement = entanglement

        # Classical head
        self.head = nn.Linear(num_wires, 1)

        # QNode
        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(x):
            # Encode data
            for i, xi in enumerate(x):
                qml.RX(xi, wires=i)
            # Variational layers
            for layer in range(layers):
                for w in range(num_wires):
                    qml.RY(self.var_params[layer, w, 0], wires=w)
                    qml.RZ(self.var_params[layer, w, 1], wires=w)
                if self.entanglement == "full":
                    for i in range(num_wires - 1):
                        qml.CNOT(wires=[i, i + 1])
                else:  # linear entanglement
                    for i in range(num_wires - 1):
                        qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.circuit = circuit

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Expectation values from the QNode
        expectations = self.circuit(state_batch)
        # Classical head
        return self.head(expectations).squeeze(-1)


__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
