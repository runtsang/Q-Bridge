"""Quantum self‑attention module built with PennyLane.

The circuit implements a parameterised rotation block followed by a chain
of CNOT gates and additional rotations to mimic the entanglement pattern
used by the classical module.  The returned expectation values of
Pauli‑Z observables form a probability‑like vector that can be fed into a
classical attention head.
"""

import pennylane as qml
import numpy as np

class QuantumSelfAttention:
    """Variational self‑attention circuit."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.wires = list(range(n_qubits))
        self.dev = qml.device("default.qubit", wires=self.wires)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.dev)
        def circuit():
            # Rotation block
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=self.wires[i])
                qml.RY(rotation_params[3 * i + 1], wires=self.wires[i])
                qml.RZ(rotation_params[3 * i + 2], wires=self.wires[i])

            # Entanglement block
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
                qml.RX(entangle_params[i], wires=self.wires[i + 1])

            # Expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the variational circuit and return a vector of expectation
        values that can be interpreted as attention scores.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape (3*n_qubits,) containing rotation angles.
        entangle_params : np.ndarray
            Array of shape (n_qubits-1,) containing entanglement rotations.
        shots : int, optional
            Number of samples for state‑vector simulation (unused for
            default device but kept for API compatibility).

        Returns
        -------
        np.ndarray
            Expectation values of shape (n_qubits,).
        """
        circuit = self._circuit(rotation_params, entangle_params)
        return np.array(circuit())

def SelfAttention():
    return QuantumSelfAttention()

__all__ = ["SelfAttention"]
