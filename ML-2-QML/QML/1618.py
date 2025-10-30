import pennylane as qml
import pennylane.numpy as np
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """
    Variational quantum classifier using Pennylane.
    Provides a QNode, encoding, weight vector sizes, and Pauli‑Z observables
    compatible with the classical build_classifier_circuit signature.
    """

    def __init__(self,
                 num_qubits: int,
                 depth: int = 2,
                 n_classes: int = 2):
        self.num_qubits = num_qubits
        self.depth = depth
        self.n_classes = n_classes

        # Parameter buffers
        self.weights = np.zeros(num_qubits * depth)
        self.encoding = list(range(num_qubits))

        self.dev = qml.device("default.qubit", wires=num_qubits)

        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qml.QNode:
        @qml.qnode(self.dev)
        def circuit(x):
            # Data encoding
            for i, xi in enumerate(x):
                qml.RX(xi, i)

            # Ansatz layers
            idx = 0
            for _ in range(self.depth):
                for i in range(self.num_qubits):
                    qml.RY(self.weights[idx], i)
                    idx += 1
                for i in range(self.num_qubits - 1):
                    qml.CZ(i, i + 1)

            # Observables
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]
        return circuit

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.circuit(x)

    def predict(self, x: np.ndarray) -> int:
        probs = self.forward(x)
        return int(np.argmax(probs))

    @classmethod
    def build_classifier_circuit(cls,
                                 num_qubits: int,
                                 depth: int = 2,
                                 n_classes: int = 2) -> Tuple[qml.QNode, Iterable, Iterable, List[qml.operation.Operation]]:
        """
        Return a Pennylane QNode and metadata matching the classical interface.

        Returns:
            circuit: QNode
            encoding: list of qubit indices used for data encoding
            weight_sizes: list of parameter counts per ansatz layer
            observables: list of PauliZ ops for each class
        """
        instance = cls(num_qubits, depth, n_classes)
        weight_sizes = [num_qubits] * depth
        observables = [qml.PauliZ(i) for i in range(n_classes)]
        return instance.circuit, instance.encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel"]
