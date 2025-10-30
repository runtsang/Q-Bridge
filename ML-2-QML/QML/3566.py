"""Hybrid quantum kernel for graph data using Pennylane."""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from typing import Sequence

__all__ = ["HybridGraphKernel", "kernel_matrix"]

class HybridGraphKernel:
    """Quantum kernel that encodes node embeddings into qubits and returns fidelity."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Encode first embedding
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            # Encode second embedding with negative angles
            for i in range(self.n_qubits):
                qml.RY(-y[i], wires=i)
            return qml.state()

        self.circuit = circuit

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the squared overlap between quantum states."""
        x = x.view(-1)
        y = y.view(-1)
        state = self.circuit(x, y)
        return torch.abs(state[0])**2

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_qubits: int = 4) -> np.ndarray:
    """Compute Gram matrix using quantum kernel."""
    kernel = HybridGraphKernel(n_qubits=n_qubits)
    return np.array([[kernel.forward(x, y).item() for y in b] for x in a])
