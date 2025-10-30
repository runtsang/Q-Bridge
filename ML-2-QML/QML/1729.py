"""Quantum kernel implementation using Pennylane.

The :class:`QuantumKernelMethod` class encodes classical data through a data‑dependent
Ry‑rotation ansatz and evaluates the quantum kernel as the squared magnitude
of the inner product of the resulting quantum states.  The implementation
focuses on clarity and provides a lightweight API that mirrors the classical
module’s ``kernel_matrix`` method.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from typing import Sequence

class QuantumKernelMethod:
    """Quantum kernel based on a fixed Ry‑rotation ansatz.

    Parameters
    ----------
    n_wires : int, default=4
        Number of qubits (features) in the data‑encoding circuit.
    device_name : str, default='default.qubit'
        Pennylane device to use.
    """

    def __init__(self, n_wires: int = 4, device_name: str = "default.qubit") -> None:
        self.n_wires = n_wires
        self.dev = qml.device(device_name, wires=n_wires)
        # Pre‑compile the state‑vector circuit
        self._circuit = qml.QNode(self._ansatz, self.dev)

    def _ansatz(self, x: np.ndarray) -> np.ndarray:
        """State‑vector circuit preparing |ψ(x)⟩."""
        for i in range(self.n_wires):
            qml.Ry(x[i], wires=i)
        return qml.statevector()

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return |⟨ψ(x)|ψ(y)⟩|² as the quantum kernel."""
        sx = self._circuit(x)
        sy = self._circuit(y)
        return abs(np.vdot(sx, sy)) ** 2

    def kernel_matrix(
        self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Compute the Gram matrix between two datasets."""
        a = np.array(a)
        b = np.array(b)
        return np.array(
            [
                [self.kernel(x, y) for y in b]
                for x in a
            ]
        )

__all__ = ["QuantumKernelMethod"]
