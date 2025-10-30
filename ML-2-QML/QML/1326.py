"""Hybrid variational classifier built with PennyLane."""

import pennylane as qml
from pennylane import numpy as np
from typing import Iterable, Tuple, List


class QuantumClassifierModel:
    """
    Variational classifier that mirrors the classical interface.
    The circuit consists of data‑encoding (RX), a depth‑wise ansatz
    of Ry + CNOT, and a set of Pauli‑Z observables for each qubit.
    """
    def __init__(
        self,
        num_qubits: int,
        depth: int = 3,
        device: qml.Device | None = None
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = device or qml.device("default.qubit", wires=num_qubits)
        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.device, interface="torch")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Data encoding
            for i, w in enumerate(inputs):
                qml.RX(w, wires=i)
            # Ansatz
            idx = 0
            for _ in range(self.depth):
                for i in range(self.num_qubits):
                    qml.RY(weights[idx], wires=i)
                    idx += 1
                for i in range(self.num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        self._circuit = circuit

    def forward(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit and return the expectation values.
        """
        return self._circuit(inputs, weights)

    def get_weight_shapes(self) -> Tuple[int,...]:
        """
        Return the shape of the weight vector for the ansatz.
        """
        return (self.num_qubits * self.depth,)

    def observables(self) -> List[qml.operation.Operator]:
        """
        Return the list of Pauli‑Z observables used in the circuit.
        """
        return [qml.PauliZ(i) for i in range(self.num_qubits)]

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int
    ) -> Tuple["QuantumClassifierModel", Iterable[int], Iterable[int], List[qml.operation.Operator]]:
        """
        Factory that returns an instance and metadata matching the classical build_classifier_circuit.
        """
        model = QuantumClassifierModel(num_qubits, depth)
        encoding = list(range(num_qubits))
        weight_sizes = [model.num_qubits * model.depth]
        observables = model.observables()
        return model, encoding, weight_sizes, observables
