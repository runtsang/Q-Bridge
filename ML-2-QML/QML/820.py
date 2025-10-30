"""
Variational quantum circuit that implements a fully‑connected layer
using a multi‑qubit parameterized circuit and a Pauli‑Z expectation.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


def FCL() -> "QuantumCircuit":
    """Return a variational quantum circuit with a single `run` method."""

    class QuantumCircuit:
        def __init__(self, n_qubits: int = 2, shots: int = 1000) -> None:
            self.n_qubits = n_qubits
            self.shots = shots
            self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
            self.qnode = qml.QNode(self._circuit, self.dev)

        def _circuit(self, thetas: np.ndarray) -> float:
            """
            Parameterized circuit: Ry rotations on each qubit followed by
            a linear chain of CNOTs. The expectation value of Pauli‑Z on
            the first qubit is returned.
            """
            for i in range(self.n_qubits):
                qml.RY(thetas[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """
            Execute the circuit with the supplied parameters and return the
            expectation value as a one‑element numpy array.
            """
            theta_arr = np.array(list(thetas), dtype=np.float64)
            if theta_arr.size!= self.n_qubits:
                raise ValueError(
                    f"Expected {self.n_qubits} parameters, got {theta_arr.size}"
                )
            expectation = self.qnode(theta_arr)
            return np.array([expectation])

    return QuantumCircuit()


__all__ = ["FCL"]
