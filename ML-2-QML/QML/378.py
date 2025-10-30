"""Quantum variational circuit that emulates a fully‑connected layer.

The circuit operates on a fixed number of qubits and returns the expectation
values of Pauli‑Z on each qubit.  Parameters are supplied via the ``thetas``
argument, allowing direct comparison with the classical MLP.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from typing import Iterable


def FCL() -> "QuantumCircuit":
    """Return a quantum circuit instance ready for execution."""
    return QuantumCircuit()


class QuantumCircuit:
    """Parameterised ansatz with Ry rotations and a single entangling layer."""

    def __init__(self, n_qubits: int = 4, device: qml.Device | None = None) -> None:
        self.n_qubits = n_qubits
        self.device = device or qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.device, interface="numpy")
        def circuit(thetas: Iterable[float]) -> np.ndarray:
            # Layer of single‑qubit rotations
            for i in range(self.n_qubits):
                qml.RY(thetas[i], wires=i)

            # Simple entanglement pattern
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Return expectation of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameter list of length *n_qubits*.

        Returns
        -------
        np.ndarray
            Expectation values of Pauli‑Z for each qubit.
        """
        thetas = list(thetas)
        if len(thetas)!= self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} parameters, got {len(thetas)}"
            )
        return np.array(self.circuit(thetas))


__all__ = ["FCL"]
