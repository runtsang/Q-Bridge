"""Quantum kernel implementation using PennyLane variational circuit."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import pennylane as qml


class HybridKernel:
    """Quantum kernel built with a PennyLane circuit."""
    def __init__(self, wires: int = 4):
        self.wires = wires
        self.dev = qml.device("default.qubit", wires=wires)

    def _circuit(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, y):
            # Encode x
            for i, val in enumerate(x):
                qml.RY(val, wires=i)
            # Entanglement
            for i in range(self.wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Encode y with negative sign
            for i, val in enumerate(y):
                qml.RY(-val, wires=i)
            # Final entanglement
            for i in range(self.wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.probs(wires=range(self.wires))
        # Return the probability of measuring all zeros
        return circuit(x, y)[0]

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._circuit(x, y)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between datasets `a` and `b` using the PennyLane quantum kernel."""
    kernel = HybridKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["HybridKernel", "kernel_matrix"]
