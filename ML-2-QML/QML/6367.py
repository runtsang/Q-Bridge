"""Hybrid quantum kernel using Pennylane.

The quantum kernel is defined as the absolute value of the overlap
between two variationally prepared states.  The ansatz is a
parameter‑shiftable circuit with a trainable rotation layer.
"""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from typing import Sequence, Callable

class QuantumKernalAnsatz:
    """Variational ansatz for data encoding.

    The circuit consists of a rotation layer that encodes the
    input vector followed by a pair of entangling layers.
    The rotation angles are trainable parameters that are
    optimized jointly with the classical RBF kernel.
    """
    def __init__(self, n_wires: int, dev: qml.Device):
        self.n_wires = n_wires
        self.dev = dev
        # Trainable rotation angles: one per wire
        self.params = torch.randn(n_wires, requires_grad=True)

    def encode(self, x: torch.Tensor) -> None:
        """Prepare a quantum state from the classical vector ``x``."""
        for i in range(self.n_wires):
            qml.RY(x[i], wires=i)
        # Entangling block
        for i in range(self.n_wires - 1):
            qml.CNOT(wires=[i, i + 1])

class Kernel:
    """Quantum kernel evaluated with Pennylane.

    The kernel is the absolute value of the inner product
    between two states prepared by the variational ansatz.
    """
    def __init__(self, n_wires: int = 4) -> None:
        self.n_wires = n_wires
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.ansatz = QuantumKernalAnsatz(n_wires, self.dev)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def _qnode(x: torch.Tensor, y: torch.Tensor):
            # Encode first vector
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i)
            # Unitary from second vector (reverse)
            for i in range(self.n_wires):
                qml.RY(-y[i], wires=i)
            return qml.state()

        self._qnode = _qnode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the quantum kernel value for a pair of feature vectors."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        state = self._qnode(x.squeeze(), y.squeeze())
        return torch.abs(state[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix using the quantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernalAnsatz", "Kernel", "kernel_matrix"]
