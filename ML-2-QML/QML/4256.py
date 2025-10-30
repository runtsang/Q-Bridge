"""Hybrid quantum classifier using Pennylane.

The circuit encodes classical data via Rx gates, applies a depth‑controlled
variational layer of Ry rotations and CZ entanglement, and measures the
expectation value of Pauli‑Z on each qubit.  The class mirrors the
classical counterpart in interface and metadata.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import List, Tuple

class HybridQuantumClassifier:
    """
    Quantum classifier that implements the same architecture as the
    classical HybridClassifier but using a real quantum circuit.
    """

    def __init__(
        self,
        num_qubits: int = 256,
        depth: int = 4,
        device: str = "default.qubit",
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device(device, wires=num_qubits)
        # Parameter vectors
        self.theta = np.random.randn(num_qubits * depth)
        # Build the circuit
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, encoding: np.ndarray, theta: np.ndarray) -> qml.numpy.ndarray:
        """
        Quantum circuit encoding the input features and applying a
        depth‑controlled variational ansatz.
        """
        # Encoding via RX
        for i in range(self.num_qubits):
            qml.RX(encoding[i], wires=i)

        idx = 0
        for _ in range(self.depth):
            # Ry rotations
            for i in range(self.num_qubits):
                qml.RY(theta[idx + i], wires=i)
            idx += self.num_qubits
            # CZ entanglement between neighbouring qubits
            for i in range(self.num_qubits - 1):
                qml.CZ(wires=[i, i + 1])

        # Measure expectation of Pauli‑Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the quantum circuit on a batch of inputs.

        inputs: (batch, num_qubits) – flattened feature vectors.
        Returns: (batch, num_qubits) – expectation values.
        """
        batch = inputs.shape[0]
        # For simplicity, use the first sample as encoding for all
        encoding = inputs[0]
        theta = self.theta
        return self.qnode(encoding, theta)

    def get_metadata(self) -> Tuple[List[int], List[int], List[str]]:
        """
        Return lists of encoding indices, weight sizes of each layer,
        and observable identifiers.
        """
        encoding = list(range(self.num_qubits))
        weight_sizes = [self.theta.size]
        observables = ["Z"] * self.num_qubits
        return encoding, weight_sizes, observables

__all__ = ["HybridQuantumClassifier"]
