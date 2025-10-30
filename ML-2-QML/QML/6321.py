"""Variational quantum classifier implemented with Pennylane."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
import pennylane.numpy as np


class QuantumClassifierModel:
    """
    Variational classifier built with Pennylane.  The class exposes a
    static ``build_classifier_circuit`` that returns a QNode, the encoding
    parameters, the trainable weights, and the list of Pauli‑Z observables
    for each qubit – matching the interface of the original Qiskit seed.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        device: str = "default.qubit",
        wires: Iterable[int] | None = None,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = qml.device(device, wires=wires or list(range(num_qubits)))
        (
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = self.build_classifier_circuit(num_qubits, depth, device, wires)

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        device: str = "default.qubit",
        wires: Iterable[int] | None = None,
    ) -> Tuple[qml.QNode, Iterable[np.ndarray], Iterable[np.ndarray], List[qml.operation.Operator]]:
        """
        Constructs a parameter‑encoded variational circuit.

        Returns
        -------
        circuit : qml.QNode
            Executable circuit that outputs expectation values of Pauli‑Z on each qubit.
        encoding : Iterable[np.ndarray]
            Parameter vector for data encoding (one angle per qubit).
        weights : Iterable[np.ndarray]
            Trainable weights for the variational layers.
        observables : List[qml.operation.Operator]
            Pauli‑Z operator for each qubit, retained for API compatibility.
        """
        wires = wires or list(range(num_qubits))
        dev = qml.device(device, wires=wires)

        @qml.qnode(dev, interface="autograd")
        def circuit(encoding: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Data encoding
            for i in range(num_qubits):
                qml.RX(encoding[i], wires=i)

            # Variational layers
            idx = 0
            for _ in range(depth):
                for i in range(num_qubits):
                    qml.RY(weights[idx], wires=i)
                    idx += 1
                for i in range(num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            # Measurement of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in wires]

        # Parameter vectors
        encoding = np.array([qml.numpy.Variable(0.0) for _ in range(num_qubits)])
        weights = np.array([qml.numpy.Variable(0.0) for _ in range(num_qubits * depth)])

        observables = [qml.PauliZ(i) for i in wires]

        return circuit, encoding, weights, observables

    def __call__(self, encoding: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return self.circuit(encoding, weights)


__all__ = ["QuantumClassifierModel"]
