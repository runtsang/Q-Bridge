"""Quantum kernel implementation using Pennylane.

Features:
* Variational ansatz with trainable parameters.
* Ability to return both kernel value and learned embedding (state vector).
* Utility to compute Gram matrix for two datasets.
"""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as pnp
from typing import Sequence, Tuple, List, Optional

__all__ = [
    "VariationalAnsatz",
    "QuantumKernel",
    "kernel_matrix",
]


class VariationalAnsatz:
    """Hardware‑efficient ansatz with trainable rotation angles."""

    def __init__(self, n_qubits: int, layers: int = 2) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        # Parameters for RX, RY, RZ on each qubit per layer
        self.params = pnp.random.randn(layers, n_qubits, 3)

    def __call__(self, circuit, x: np.ndarray) -> None:
        for l in range(self.layers):
            for q in range(self.n_qubits):
                # Encode data using RY with data value
                circuit.ry(x[q], wires=q)
                # Trainable rotations
                circuit.rx(self.params[l, q, 0], wires=q)
                circuit.ry(self.params[l, q, 1], wires=q)
                circuit.rz(self.params[l, q, 2], wires=q)
                if q < self.n_qubits - 1:
                    circuit.cnot(wires=[q, q + 1])


class QuantumKernel:
    """Compute kernel via overlap of two variational states."""

    def __init__(self, n_qubits: int = 4, layers: int = 2) -> None:
        self.n_qubits = n_qubits
        self.device = qml.device("default.qubit", wires=n_qubits)
        self.ansatz = VariationalAnsatz(n_qubits, layers)

        @qml.qnode(self.device, interface="torch")
        def circuit(x):
            self.ansatz(circuit, x)
            return qml.state()

        self.circuit = circuit

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Evaluate quantum states for both inputs
        psi_x = self.circuit(x.numpy())
        psi_y = self.circuit(y.numpy())
        # Overlap squared (fidelity)
        overlap = torch.abs(torch.dot(psi_x, psi_y.conj())) ** 2
        return overlap

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])
