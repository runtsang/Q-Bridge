"""Quantum kernel using a parameter‑shift variational circuit and state‑vector overlap."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml

class QuantumKernelMethod:
    """Quantum kernel based on state‑vector overlap.

    The encoding circuit prepares a state |ψ(x)⟩ by applying a
    series of RY rotations (parameterised by the data vector x) and an
    entanglement layer.  The kernel value is |⟨ψ(x)|ψ(y)⟩|², which
    is estimated by explicitly computing the state‑vectors for *x*
    and *y* and taking the absolute square of their dot product.
    """

    def __init__(self, n_qubits: int = 4, dev: qml.Device | None = None):
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

    def _statevector(self, data: np.ndarray) -> np.ndarray:
        """Return the state‑vector produced by the encoding circuit."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            for i in range(self.n_qubits):
                qml.RY(data[i], wires=i)
            # Entanglement (linear chain)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.state()

        return circuit().numpy()

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the kernel value |⟨ψ(x)|ψ(y)⟩|²."""
        svx = self._statevector(x)
        svy = self._statevector(y)
        return float(np.abs(np.vdot(svx, svy)) ** 2)

    @staticmethod
    def kernel_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        """Compute the Gram matrix for two collections of data vectors.

        Parameters
        ----------
        a, b : sequences of 1‑D arrays
            Data points for which the kernel matrix is required.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (len(a), len(b)).
        """
        kernel = QuantumKernelMethod()
        return np.array([[kernel.kernel(x, y) for y in b] for x in a])

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
