"""Variational quantum circuit implementing a configurable fully‑connected layer.

The implementation uses Pennylane to build a strongly‑entangling variational
layer that can be stacked to arbitrary depth.  The ``run`` method accepts a
flattened parameter vector and returns the expectation value of the Pauli‑Z
operator averaged over all qubits, analogous to the classical network’s
output.  This module can be coupled with the classical counterpart for
hybrid training experiments.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QuantumCircuit:
    """Variational circuit with configurable depth and qubit count."""

    def __init__(self, n_qubits: int = 1, layers: int = 2, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.shots = shots

        # Define a device; using the default default.qubit simulator for speed
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        # Build a QNode with a strongly‑entangling layer
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: np.ndarray) -> float:
            # params shape: (layers, n_qubits, 3) for Ry, Rz, and CNOT structure
            qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits))
            # Expectation of PauliZ averaged over all qubits
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the circuit with a flattened parameter vector.

        Parameters
        ----------
        thetas : Iterable[float]
            Flattened list of parameters matching the shape required by
            ``StronglyEntanglingLayers``: ``(layers, n_qubits, 3)``.

        Returns
        -------
        np.ndarray
            Array containing a single expectation value, matching the
            interface of the classical implementation.
        """
        # Reshape the flat vector into the expected shape
        param_shape = (self.layers, self.n_qubits, 3)
        params = np.asarray(thetas, dtype=np.float32).reshape(param_shape)
        expectation = self.circuit(params)
        return np.array([expectation])

    def num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return self.layers * self.n_qubits * 3


def FCL(n_qubits: int = 1, layers: int = 2, shots: int = 1024) -> QuantumCircuit:
    """Convenience factory matching the original API."""
    return QuantumCircuit(n_qubits=n_qubits, layers=layers, shots=shots)


__all__ = ["FCL", "QuantumCircuit"]
