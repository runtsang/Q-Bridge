"""Quantum RBF kernel using PennyLane with a variational ansatz."""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from typing import Sequence

class RBFKernel:
    """Quantum kernel that evaluates the overlap of data‑encoded states."""
    def __init__(self, n_wires: int = 4, device_name: str = "default.qubit", shots: int = 1000) -> None:
        self.n_wires = n_wires
        self.dev = qml.device(device_name, wires=n_wires, shots=shots)
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, y):
            # Encode first datum
            for i, xi in enumerate(x):
                qml.RY(xi, wires=i)
            # Encode second datum with opposite sign
            for i, yi in enumerate(y):
                qml.RY(-yi, wires=i)
            # Entangling layer
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        return circuit

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Accept 1‑D tensors of length n_wires
        x = x.reshape(-1, self.n_wires)
        y = y.reshape(-1, self.n_wires)
        assert x.shape == y.shape, "x and y must have the same shape"
        return torch.abs(self.circuit(x, y))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two collections of feature vectors using the quantum kernel."""
    kernel = RBFKernel()
    A = torch.vstack([t.reshape(-1, t.shape[-1]) for t in a])
    B = torch.vstack([t.reshape(-1, t.shape[-1]) for t in b])
    return np.array([[kernel(a_i, b_j).item() for b_j in b] for a_i in a])

__all__ = ["RBFKernel"]
