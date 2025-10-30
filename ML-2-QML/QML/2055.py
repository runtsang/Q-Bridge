"""
Quantum fully‑connected layer implemented with Pennylane.
The layer is a parameterised circuit that maps a single parameter
to the expectation value of Pauli‑Z on the last qubit.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Iterable


class FCL:
    """
    Variational quantum circuit acting as a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit. Must be >= 1.
    shots : int, default 1000
        Number of measurement shots for expectation estimation.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 1000) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # Apply a rotation to each qubit
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            # Entangle all qubits in a chain
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of Pauli‑Z on the last qubit
            return qml.expval(qml.PauliZ(wires=n_qubits - 1))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for a list of parameter vectors.

        Parameters
        ----------
        thetas : Iterable[Iterable[float]]
            Each inner iterable must contain `n_qubits` parameters.

        Returns
        -------
        np.ndarray
            Array of expectation values, one per input parameter set.
        """
        # Pad or truncate each theta vector to match n_qubits
        processed = [
            np.array(thetas_i, dtype=np.float64)[: self.n_qubits]
            if len(thetas_i) >= self.n_qubits
            else np.pad(np.array(thetas_i, dtype=np.float64), (0, self.n_qubits - len(thetas_i)), "constant")
            for thetas_i in thetas
        ]
        results = [self.circuit(pnp.array(t)) for t in processed]
        return np.array(results, dtype=np.float64)

__all__ = ["FCL"]
