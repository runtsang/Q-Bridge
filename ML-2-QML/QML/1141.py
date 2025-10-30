"""Quantum variational fully‑connected layer using Pennylane.

This implementation expands the original single‑qubit circuit to a
parameterized ansatz on *n_qubits* with entangling CNOT layers.
The `run` method evaluates the expectation value of Pauli‑Z on each
qubit, returning a vector of outputs that can be fed into a classical
pipeline.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pennylane as qml
import numpy as np


class FCL:
    """
    Variational quantum circuit acting as a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the ansatz.
    dev_name : str, default "default.qubit"
        Pennylane device name.
    shots : int, default 1024
        Number of shots for expectation estimation.
    """

    def __init__(self, n_qubits: int = 1, dev_name: str = "default.qubit", shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)
        self.params = np.zeros(n_qubits * 3)  # 3 rotations per qubit

        @qml.qnode(self.dev)
        def circuit(theta):
            # Parameterized rotations
            for i in range(self.n_qubits):
                qml.RY(theta[i], wires=i)
                qml.RZ(theta[i + self.n_qubits], wires=i)
                qml.RX(theta[i + 2 * self.n_qubits], wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit for a given set of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameters for the rotations. Length must be
            `3 * n_qubits`; excess values are ignored.

        Returns
        -------
        np.ndarray
            Expectation values of Pauli‑Z for each qubit.
        """
        theta_arr = np.asarray(list(thetas), dtype=np.float64)
        if theta_arr.size < self.n_qubits * 3:
            raise ValueError(
                f"Expected at least {self.n_qubits * 3} parameters, got {theta_arr.size}"
            )
        # Truncate to required length
        theta_arr = theta_arr[: self.n_qubits * 3]
        self.params = theta_arr
        return np.array(self._circuit(theta_arr))

__all__ = ["FCL"]
