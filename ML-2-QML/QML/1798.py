"""Quantum classifier using a PennyLane variational circuit with data encoding
and Pauli‑Z observables.  The class mirrors the classical interface and
provides a ``get_metadata`` method for compatibility with hybrid loss
implementations."""
import pennylane as qml
import numpy as np
from typing import Iterable, Tuple


class QuantumClassifierModel:
    """Builds a parametric quantum circuit for binary classification.

    The circuit includes:
    * Data encoding via RX gates
    * `depth` variational layers of RY rotations and CZ entanglement
    * Pauli‑Z observables on each qubit

    The ``forward`` method returns the expectation values of the
    observables, which can be used as quantum logits.
    """
    def __init__(
        self,
        num_qubits: int,
        depth: int,
        device: qml.Device | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = device or qml.device("default.qubit", wires=num_qubits)

        # Encode the qubit indices – identical to the seed
        self.encoding: list[str] = [f"x_{i}" for i in range(num_qubits)]

        # Random initial weights
        self.weights: np.ndarray = np.random.randn(num_qubits * depth)

        self._build_circuit()

        # Pauli‑Z observables for each qubit
        self.observables: list[qml.operation.Operator] = [
            qml.PauliZ(i) for i in range(num_qubits)
        ]

    def _build_circuit(self) -> None:
        """Create a PennyLane QNode that implements the variational ansatz."""
        @qml.qnode(self.device)
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> list[float]:
            # Data encoding
            for w, q in zip(inputs, range(self.num_qubits)):
                qml.RX(w, q)

            # Variational layers
            idx = 0
            for _ in range(self.depth):
                for q in range(self.num_qubits):
                    qml.RY(weights[idx], q)
                    idx += 1
                for q in range(self.num_qubits - 1):
                    qml.CZ(q, q + 1)

            # Return expectation values of the observables
            return [qml.expval(obs) for obs in self.observables]

        self.circuit = circuit

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the circuit on input `x`."""
        return np.array(self.circuit(x, self.weights))

    def get_metadata(
        self,
    ) -> Tuple[qml.QNode, Iterable[str], np.ndarray, list[qml.operation.Operator]]:
        """Return the QNode and metadata matching the seed's tuple."""
        return self.circuit, self.encoding, self.weights, self.observables
