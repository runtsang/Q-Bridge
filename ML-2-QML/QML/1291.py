"""Quantum kernel using a parameterised variational circuit with PennyLane."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as pnp


class KernalAnsatz:
    """Variational ansatz for quantum kernel."""
    def __init__(self, n_wires: int = 4, depth: int = 2):
        self.n_wires = n_wires
        self.depth = depth

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single feature vector into a quantum state."""
        dev = qml.device("default.qubit", wires=self.n_wires)

        @qml.qnode(dev, interface="torch")
        def circuit(x):
            # Data encoding with RX rotations
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i)
            # Variational entangling layers
            for _ in range(self.depth):
                for i in range(self.n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(self.n_wires):
                    qml.RY(0.0, wires=i)  # placeholder for trainable params
            return qml.state()

        return circuit(x)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the squared overlap |⟨ψ_x|ψ_y⟩|² as the kernel value."""
        state_x = self.encode(x)
        state_y = self.encode(y)
        overlap = torch.dot(state_x.conj(), state_y)
        return torch.abs(overlap) ** 2


class Kernel:
    """Quantum kernel evaluated via a fixed variational ansatz."""
    def __init__(self, n_wires: int = 4, depth: int = 2):
        self.ansatz = KernalAnsatz(n_wires, depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel()
    return np.array([[kernel.forward(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
