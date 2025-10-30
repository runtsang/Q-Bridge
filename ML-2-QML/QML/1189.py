"""Variational quantum self‑attention using PennyLane."""

from __future__ import annotations

import numpy as np
import pennylane as qml


def SelfAttention():
    class QuantumSelfAttention:
        """
        Variational circuit that produces attention scores as
        Pauli‑Z expectation values. The circuit is parameterized by
        rotation and entanglement angles supplied at runtime.
        """

        def __init__(self, n_qubits: int = 4):
            self.n_qubits = n_qubits
            self.wires = list(range(n_qubits))
            # Default device; can be overridden in run()
            self.dev = qml.device("default.qubit", wires=self.wires)

        def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
            # Single‑qubit rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=self.wires[i])
                qml.RY(rotation_params[3 * i + 1], wires=self.wires[i])
                qml.RZ(rotation_params[3 * i + 2], wires=self.wires[i])

            # Entanglement layer (CNOT chain)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])

            # Return expectation values of Pauli‑Z for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in self.wires]

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024,
        ) -> np.ndarray:
            """
            Execute the variational circuit and return attention scores.

            Parameters
            ----------
            rotation_params : np.ndarray
                Shape (3 * n_qubits,) – rotation angles for RX, RY, RZ.
            entangle_params : np.ndarray
                Shape (n_qubits - 1,) – unused in this simple design but kept for API compatibility.
            shots : int
                Number of measurement shots.

            Returns
            -------
            np.ndarray
                Attention scores, one per qubit, shape (n_qubits,).
            """
            # Re‑initialize device with shots
            self.dev = qml.device("default.qubit", wires=self.wires, shots=shots)
            qnode = qml.QNode(self._circuit, self.dev)
            return qnode(rotation_params, entangle_params)

    return QuantumSelfAttention(n_qubits=4)


__all__ = ["SelfAttention"]
