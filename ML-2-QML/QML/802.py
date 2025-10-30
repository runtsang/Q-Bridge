"""Quantum fully‑connected layer using Pennylane.

The module defines a variational circuit that operates on a single qubit
with a depth‑2 entanglement pattern.  The ``run`` method accepts a list
of parameters and returns the expectation value of Pauli‑Z on the
first qubit.  The implementation is fully compatible with the
original ``FCL`` API while enabling quantum‑enhanced experiments.

Typical usage::

    qcl = FCL()
    expectation = qcl.run([0.1, 0.2, 0.3, 0.4])
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pennylane as qml
import numpy as np


class QuantumFullyConnectedLayer:
    """
    Variational circuit that emulates a fully‑connected layer.

    The circuit consists of two layers of parameterized rotations
    followed by a CNOT entanglement.  The parameters are supplied via
    the ``run`` method.  The circuit returns the expectation value
    of the Pauli‑Z observable on the first qubit.
    """

    def __init__(self, n_qubits: int = 1, device_name: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: Sequence[float]) -> float:
            # Ensure we have enough parameters
            if len(params) < 4 * self.n_qubits:
                # Pad with zeros
                params = np.pad(params, (0, 4 * self.n_qubits - len(params)), "constant")

            # Layer 1: Rotation-RY and entanglement
            for w in range(self.n_qubits):
                qml.RY(params[4 * w + 0], wires=w)
            for w in range(self.n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])

            # Layer 2: Rotation-RZ and entanglement
            for w in range(self.n_qubits):
                qml.RZ(params[4 * w + 1], wires=w)
            for w in range(self.n_qubits - 1):
                qml.CNOT(wires=[w + 1, w])

            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the variational circuit with the supplied parameters.

        Parameters
        ----------
        thetas:
            Iterable of floats that will be bound to the circuit
            parameters.  The list is flattened and padded if needed.

        Returns
        -------
        np.ndarray
            A 1‑D array containing the expectation value of Pauli‑Z.
        """
        expectation = self.circuit(np.array(list(thetas), dtype=np.float64))
        return np.array([expectation])


def FCL(n_qubits: int = 1, device_name: str = "default.qubit") -> QuantumFullyConnectedLayer:
    """
    Factory function that returns an instance of ``QuantumFullyConnectedLayer``.
    Mirrors the original API but offers a richer quantum circuit.

    Returns
    -------
    QuantumFullyConnectedLayer
    """
    return QuantumFullyConnectedLayer(n_qubits=n_qubits, device_name=device_name)


__all__ = ["FCL"]
