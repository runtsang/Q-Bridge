"""Quantum regression model using Pennylane with tunable encoding and richer feature extraction."""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as pnp
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate quantum states of the form cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩.
    The target is sin(2θ) * cos(φ).
    """
    rng = np.random.default_rng()
    thetas = rng.uniform(0, 2 * np.pi, size=samples)
    phis = rng.uniform(0, 2 * np.pi, size=samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset yielding quantum state vectors and targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegression__gen019(nn.Module):
    """
    Variational quantum circuit with tunable encoding and Pauli‑X/Y measurement
    followed by a classical linear head.
    """
    def __init__(self, num_wires: int, encoding: str = "ry", num_layers: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.encoding = encoding
        self.num_layers = num_layers

        # Device for Pennylane
        self.dev = qml.device("default.qubit", wires=num_wires)

        # Variational parameters
        self.params = nn.Parameter(torch.randn(num_layers * 3 * num_wires))

        # Classical head
        self.head = nn.Linear(num_wires * 2, 1)  # *2 for X and Y features

    def _encoding(self, x: torch.Tensor):
        """
        Apply the chosen encoding to the quantum device.
        """
        if self.encoding == "ry":
            for i in range(self.num_wires):
                qml.RY(x[i], wires=i)
        elif self.encoding == "rz":
            for i in range(self.num_wires):
                qml.RZ(x[i], wires=i)
        elif self.encoding == "rx":
            for i in range(self.num_wires):
                qml.RX(x[i], wires=i)
        else:
            raise ValueError(f"Unsupported encoding: {self.encoding}")

    def _variational_layer(self, layer_idx: int):
        """
        Apply a single variational layer with rotation gates.
        """
        offset = layer_idx * 3 * self.num_wires
        for i in range(self.num_wires):
            idx = offset + 3 * i
            qml.Rot(self.params[idx], self.params[idx + 1], self.params[idx + 2], wires=i)

    def _entangle(self):
        """
        Simple linear entanglement pattern.
        """
        for i in range(self.num_wires - 1):
            qml.CNOT(wires=[i, i + 1])

    def circuit(self, x, params):
        """
        Pennylane circuit definition. Uses interface='torch' to return torch tensors.
        """
        self._encoding(x)
        for l in range(self.num_layers):
            self._variational_layer(l)
            self._entangle()
        # Measure both Pauli‑Z and Pauli‑X on each wire
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)] + \
               [qml.expval(qml.PauliX(i)) for i in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: evaluate the circuit for each batch element and feed into the linear head.
        """
        batch_features = []
        for xi in state_batch:
            feat = qml.QNode(self.circuit, self.dev, interface="torch")(xi, self.params)
            batch_features.append(feat)
        batch_features = torch.stack(batch_features)
        return self.head(batch_features).squeeze(-1)


__all__ = ["QuantumRegression__gen019", "RegressionDataset", "generate_superposition_data"]
