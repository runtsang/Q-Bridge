"""Quantum classifier implementation using Pennylane."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import Expectation


class QuantumClassifierModel:
    """
    Variational quantum classifier that mirrors the classical API.
    The circuit consists of data encoding via RX, followed by
    multiple layers of Ry rotations and CZ entanglement.
    """

    def __init__(self, num_qubits: int, depth: int):
        self.num_qubits = num_qubits
        self.depth = depth
        self._build_circuit()

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int, depth: int
    ) -> Tuple[qml.QuantumNode, Iterable, Iterable, List[Expectation]]:
        """
        Static helper that returns the quantum node, encoding parameters,
        variational parameters and measurement operators.
        """
        # Symbolic parameters
        encoding = [qml.numpy.Variable(f"x{q}") for q in range(num_qubits)]
        theta = [
            qml.numpy.Variable(f"theta_{l}_{q}") for l in range(depth) for q in range(num_qubits)
        ]

        @qml.qnode(qml.device("default.qubit", wires=num_qubits))
        def circuit(inputs: np.ndarray):
            # Data encoding
            for q, x in enumerate(inputs):
                qml.RX(x, wires=q)

            # Variational layers
            idx = 0
            for _ in range(depth):
                for q in range(num_qubits):
                    qml.RY(theta[idx + q], wires=q)
                for q in range(num_qubits - 1):
                    qml.CZ(wires=[q, q + 1])
                idx += num_qubits

            # Return expectation values of Z on each qubit
            return [qml.expval(qml.Z(w)) for w in range(num_qubits)]

        observables = [qml.expval(qml.Z(w)) for w in range(num_qubits)]
        return circuit, encoding, theta, observables

    def _build_circuit(self):
        self.circuit, self.encoding, self.weights, self.observables = (
            self.build_classifier_circuit(self.num_qubits, self.depth)
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum circuit on the provided data point.
        """
        return self.circuit(x)

__all__ = ["QuantumClassifierModel"]
