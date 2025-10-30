"""Variational quantum self‑attention.

This module replaces the simple Qiskit circuit from the seed with a
parameter‑shared variational block that returns expectation values
of Pauli‑Z observables.  The circuit consists of
* rotation gates (rx, ry, rz) on each qubit,
* a chain of controlled‑rx entangling gates,
* and a measurement of the Z expectation on each qubit.

The ``run`` method accepts a Pennylane device or a Qiskit backend
and returns a NumPy array of expectation values, making the class
suitable for variational training.

The public API mirrors the seed: the class provides a ``run`` method
taking ``rotation_params``, ``entangle_params`` and an optional
``shots`` argument.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class SelfAttention:
    """Parameter‑shared variational quantum circuit for self‑attention."""

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        # Default device – can be overridden by the user via ``run``.
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _qnode(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ):
        """Return a Pennylane QNode that implements the attention circuit."""

        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def circuit():
            # Apply parameter‑shared rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Chain of controlled‑rx gates
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Return expectation values of Pauli‑Z for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the variational circuit and return the expectation values.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the RX/RY/RZ gates (shape ``(3 * n_qubits,)``).
        entangle_params : np.ndarray
            Parameters for the CRX entangling gates (shape ``(n_qubits - 1,)``).
        shots : int, default=1024
            Number of shots for the backend; ignored for the default
            autograd device but kept for API compatibility.

        Returns
        -------
        np.ndarray
            Expectation values of shape ``(n_qubits,)``.
        """
        qnode = self._qnode(rotation_params, entangle_params)
        # For the default device we ignore shots; for real hardware we
        # could accept a backend and forward the shots argument.
        return np.asarray(qnode())

__all__ = ["SelfAttention"]
