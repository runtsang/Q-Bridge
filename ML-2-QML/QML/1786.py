"""Quantum implementation of a fully‑connected layer via a parameterized circuit.

Each input feature corresponds to a qubit. The circuit consists of a layer of Ry rotations,
entangling CNOTs, and a second Ry layer. The expectation value of the Z measurement on the
first qubit is returned as the layer output.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


class FullyConnectedLayer:
    """
    Quantum variational circuit that emulates a neural network layer.
    """

    def __init__(self, n_qubits: int = 1, device: str = "default.qubit", shots: int = 1000):
        self.n_qubits = n_qubits
        self.device = qml.device(device, wires=n_qubits, shots=shots)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device)
        def circuit(params):
            # params shape: (n_qubits, 3)
            for i in range(self.n_qubits):
                qml.RY(params[i, 0], wires=i)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            for i in range(self.n_qubits):
                qml.RY(params[i, 1], wires=i)
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        thetas : np.ndarray
            Flattened array of parameters of shape (n_qubits, 3).

        Returns
        -------
        np.ndarray
            Expectation value of the first qubit's Z operator.
        """
        params = thetas.reshape(self.n_qubits, 3)
        expectation = self.circuit(params)
        return np.array([expectation])

    def __call__(self, thetas: np.ndarray) -> np.ndarray:
        return self.run(thetas)


def FCL(n_qubits: int = 1, device: str = "default.qubit", shots: int = 1000):
    """Convenience factory matching the original seed API."""
    return FullyConnectedLayer(n_qubits, device, shots)


__all__ = ["FullyConnectedLayer", "FCL"]
