"""Quantum kernel construction using PennyLane variational ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class KernalAnsatz:
    """Parameterized quantum kernel ansatz.

    The ansatz encodes two classical data vectors into a quantum state
    by applying a trainable rotation on each qubit followed by a
    layer of CNOTs. The kernel value is the absolute overlap between
    the states prepared from ``x`` and ``-y``.
    """

    def __init__(self, wires: int, dev: qml.Device | None = None):
        self.wires = wires
        self.dev = dev or qml.device("default.qubit", wires=wires)
        # Trainable parameters for the rotation layer
        self.params = pnp.random.randn(2 * wires, requires_grad=True)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, y, params):
            # Encode x
            for i in range(self.wires):
                qml.RY(x[i], wires=i)
            # Apply trainable rotations
            for i in range(self.wires):
                qml.RZ(params[i], wires=i)
            # Entangling layer
            for i in range(self.wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Encode -y (reverse sign)
            for i in range(self.wires):
                qml.RY(-y[i], wires=i)
            # Apply inverse trainable rotations
            for i in range(self.wires):
                qml.RZ(-params[self.wires + i], wires=i)
            return qml.state()

        self.circuit = circuit

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the absolute overlap of the two encoded states."""
        psi_x = self.circuit(x, y, self.params)
        psi_y = self.circuit(y, x, self.params)
        overlap = np.abs(np.vdot(psi_x, psi_y))
        return float(overlap)

class Kernel:
    """Quantum kernel that wraps :class:`KernalAnsatz`."""

    def __init__(self, wires: int = 4):
        self.ansatz = KernalAnsatz(wires)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.ansatz(x, y)

def kernel_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
    """Compute the Gram matrix between two collections of samples.

    Parameters
    ----------
    a, b : Sequence[np.ndarray]
        Each element should be a 1‑D array of length ``wires``.
    """
    kernel = Kernel()
    K = np.array([[kernel(x, y) for y in b] for x in a])
    return K

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
