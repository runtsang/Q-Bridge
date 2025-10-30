"""Quantum self‑attention module using Pennylane.

This module implements a parameterized quantum circuit that outputs a
probability distribution over 2ⁿ qubit basis states.  The distribution
is used as a mask in the hybrid attention module.
"""

import numpy as np
import pennylane as qml

class SelfAttention__gen226:
    """Quantum self‑attention that returns a probability mask."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Parameterized circuit with single‑qubit rotations and controlled‑RZ gates."""
        for i in range(self.n_qubits):
            qml.RX(rotation_params[3 * i], wires=i)
            qml.RY(rotation_params[3 * i + 1], wires=i)
            qml.RZ(rotation_params[3 * i + 2], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])
        return [qml.probs(wires=range(self.n_qubits))]

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        """Execute the circuit and return the probability distribution."""
        probs = qml.execute([self._circuit], self.dev, [rotation_params, entangle_params])
        return probs[0]

__all__ = ["SelfAttention__gen226"]
