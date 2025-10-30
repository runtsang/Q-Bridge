"""Quantum kernel module using a variational ansatz.

Key extensions:
- The ansatz is fully trainable; parameters are stored as a PyTorch ``Parameter``.
- Kernel is defined as the squared fidelity between two quantum states.
- A helper ``kernel_matrix`` produces the Gram matrix for arbitrary data sets.
- Uses PennyLane's ``default.qubit`` simulator for fast CPU evaluation.

The public names ``KernalAnsatz``, ``Kernel`` and ``kernel_matrix`` are kept for backward
compatibility, while ``QuantumKernelMethod`` is the new, user‑facing class.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """Variational ansatz that maps classical data into a quantum state."""

    def __init__(self, num_qubits: int, layers: int = 2) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        # Trainable rotation angles: shape (layers, num_qubits, 3) for RX, RY, RZ
        self.params = nn.Parameter(torch.randn(layers, num_qubits, 3))

    def _variational_circuit(self, data: torch.Tensor) -> torch.Tensor:
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor):
            # Data encoding
            for q in range(self.num_qubits):
                qml.RY(x[q], wires=q)
            # Variational layers
            for l in range(self.layers):
                for q in range(self.num_qubits):
                    qml.RX(self.params[l, q, 0], wires=q)
                    qml.RY(self.params[l, q, 1], wires=q)
                    qml.RZ(self.params[l, q, 2], wires=q)
                # Entangling layer (ring of CNOTs)
                for q in range(self.num_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.num_qubits])
            return qml.state()
        return circuit(data)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        psi_x = self._variational_circuit(x)
        psi_y = self._variational_circuit(y)
        fidelity = torch.abs(torch.dot(psi_x.conj(), psi_y)) ** 2
        return fidelity


class Kernel(nn.Module):
    """Quantum kernel wrapper that mirrors the classical API."""

    def __init__(self, num_qubits: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(num_qubits, layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self(x, y).item() for y in b] for x in a])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class QuantumKernelMethod:
    """Convenience wrapper that mirrors the original API."""

    def __init__(self, num_qubits: int = 4, layers: int = 2) -> None:
        self.kernel = Kernel(num_qubits, layers)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return self.kernel.kernel_matrix(a, b)


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "QuantumKernelMethod"]
