"""QuantumClassifier implemented with Pennylane variational circuit."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
import pennylane.numpy as np
import torch
from qiskit.quantum_info import SparsePauliOp  # for observables consistency


class QuantumClassifier:
    """
    Variational quantum circuit with data‑re‑uploading and multi‑qubit entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features are encoded one‑to‑one).
    depth : int, default=3
        Number of variational layers.

    The circuit is wrapped in a QNode that accepts a PyTorch tensor of shape
    (batch_size, num_qubits) and a parameter vector of shape
    (num_qubits * depth,).  Metadata mirrors the classical API.
    """

    def __init__(self, num_qubits: int, depth: int = 3) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = list(range(num_qubits))
        self.weights = np.arange(num_qubits * depth, dtype=np.float32)
        self.observables: List[SparsePauliOp] = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        self.device = qml.device("default.qubit", wires=num_qubits)
        self.weight_sizes: List[int] = [num_qubits * depth]

        @qml.qnode(self.device, interface="torch")
        def circuit(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            """Data‑re‑uploading ansatz."""
            # Encoding
            for i, qubit in enumerate(range(num_qubits)):
                qml.RX(x[:, i], wires=qubit)

            # Variational layers
            idx = 0
            for _ in range(depth):
                for qubit in range(num_qubits):
                    qml.RY(theta[idx], wires=qubit)
                    idx += 1
                # Entangle with CZ ladder
                for qubit in range(num_qubits - 1):
                    qml.CZ(qubit, qubit + 1)

            # Measurement of Pauli‑Z on each qubit
            return [qml.expval(obs) for obs in self.observables]

        self.circuit = circuit
        self.model = self.circuit

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the circuit."""
        return self.circuit(x, theta)

    def get_metadata(self) -> Tuple[torch.nn.Module, Iterable[int], Iterable[int], List[SparsePauliOp]]:
        """
        Return the QNode and metadata for compatibility with the classical API.
        """
        return self.model, self.encoding, self.weight_sizes, self.observables


__all__ = ["QuantumClassifier"]
