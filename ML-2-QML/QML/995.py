"""Quantum classifier using Pennylane.

The class builds a variational circuit with data‑encoding, a depth‑controlled ansatz,
and a set of Z‑observables.  The QNode is fully differentiable and can be
optimised with any PyTorch optimiser thanks to Pennylane’s autograd bridge."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
import pennylane.numpy as np  # NumPy‑like API with autograd support


class QuantumClassifierModel:
    """Variational quantum classifier with configurable depth and device."""

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        device: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        """
        Parameters
        ----------
        num_qubits:
            Number of qubits / input features.
        depth:
            Number of ansatz layers.
        device:
            Pennylane device name.
        shots:
            Number of measurement shots for expectation estimation.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = qml.device(device, wires=num_qubits, shots=shots)

        # Parameter vectors
        self.encoding_params = np.arange(num_qubits, dtype=np.float32)
        self.ansatz_params = np.arange(num_qubits * depth, dtype=np.float32)

        # Observables: single‑qubit Z on each wire
        self.observables = [
            qml.PauliZ(wire) for wire in range(num_qubits)
        ]

        # Build the QNode
        @qml.qnode(self.device, interface="torch")
        def circuit(params, encoding):
            # Data encoding
            for i, wire in enumerate(range(num_qubits)):
                qml.RX(encoding[i], wire)

            # Ansatz layers
            idx = 0
            for _ in range(depth):
                for wire in range(num_qubits):
                    qml.RY(params[idx], wire)
                    idx += 1
                # Entangling CZ chain
                for wire in range(num_qubits - 1):
                    qml.CZ(wire, wire + 1)

        self.circuit = circuit

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Evaluate the circuit and return expectation values."""
        return self.circuit(self.ansatz_params, data)

    def loss(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Binary cross‑entropy loss for the classifier."""
        preds = self.__call__(data)
        # Convert Z‑expectations to probabilities via (1 + exp(-2x))/2
        probs = 0.5 * (1 + np.exp(-2 * preds))
        # Binary cross‑entropy
        eps = 1e-12
        return -np.mean(
            labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)
        )

    def get_params(self) -> List[np.ndarray]:
        """Return the ansatz parameters as a list for optimisation."""
        return [self.ansatz_params]

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        device: str = "default.qubit",
        shots: int = 1024,
    ) -> Tuple["QuantumClassifierModel", Iterable, Iterable, List[qml.PauliZ]]:
        """
        Factory method mirroring the original signature but returning the
        Pennylane model and metadata.
        """
        model = QuantumClassifierModel(num_qubits, depth, device=device, shots=shots)
        return model, model.encoding_params, model.ansatz_params, model.observables


__all__ = ["QuantumClassifierModel"]
