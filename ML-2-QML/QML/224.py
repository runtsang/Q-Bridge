"""Quantum regression model using PennyLane.

The implementation introduces amplitude encoding of the classical data,
a parameterized variational ansatz with entangling CNOT layers,
and a classical linear head.  The circuit is defined as a PennyLane
QNode with Torch interface, enabling end‑to‑end differentiation
on GPU.  The model is compatible with the same dataset as the
classical version.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
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


class RegressionDataset(Dataset):
    """
    PyTorch Dataset wrapping the synthetic quantum regression data.
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


def _create_ansatz(num_wires: int, n_layers: int):
    """Return a PennyLane QNode that applies amplitude encoding and a
    parameterized variational circuit with entangling CNOTs."""
    dev = qml.device("default.qubit", wires=num_wires)

    @qml.qnode(dev, interface="torch")
    def circuit(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # amplitude encoding via Ry rotations (linear mapping)
        for i, val in enumerate(x):
            qml.RY(val, wires=i)
        # variational layers
        for layer in range(n_layers):
            for wire in range(num_wires):
                qml.RX(params[layer, wire, 0], wires=wire)
                qml.RY(params[layer, wire, 1], wires=wire)
                qml.RZ(params[layer, wire, 2], wires=wire)
            # nearest‑neighbour entanglement
            for wire in range(num_wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

    return circuit


class QModel(nn.Module):
    """
    Hybrid quantum‑classical regression model.
    """
    def __init__(self, num_wires: int, n_layers: int = 2, dev_type: str = "default.qubit"):
        super().__init__()
        self.num_wires = num_wires
        self.n_layers = n_layers
        self.circuit = _create_ansatz(num_wires, n_layers)
        # Trainable parameters of the variational ansatz
        self.ansatz_params = nn.Parameter(torch.randn(n_layers, num_wires, 3))
        # Classical linear head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Evaluate the circuit for each sample in the batch
        features = []
        for x in state_batch:
            features.append(self.circuit(x, self.ansatz_params))
        features = torch.stack(features)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
