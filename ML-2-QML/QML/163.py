"""Variational quantum classifier built with Pennylane."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
import numpy as np

class QuantumClassifier:
    """
    Variational circuit with data‑encoding and multi‑observable readouts.
    Parameters:
        num_qubits: number of data‑encoded qubits.
        depth: number of variational layers.
        device: PennyLane device; defaults to ``default.qubit``.
    """
    def __init__(self, num_qubits: int, depth: int,
                 device: qml.Device | None = None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = device or qml.device("default.qubit", wires=num_qubits, shots=1024)
        # Initialise variational parameters randomly
        self.variational_params = np.random.uniform(0, 2 * np.pi,
                                                    size=(depth, num_qubits))
        self.encoding_params = np.zeros(num_qubits)
        self._build_qnode()

    def _ansatz(self, data: np.ndarray, params: np.ndarray):
        """Data‑encoding followed by variational layers."""
        # Angle‑encoding of data
        for i, wire in enumerate(range(self.num_qubits)):
            qml.RX(data[i], wires=wire)
        # Variational layers
        for layer, layer_params in zip(range(self.depth), params):
            for q in range(self.num_qubits):
                qml.RY(layer_params[q], wires=q)
            # Entanglement
            for q in range(self.num_qubits - 1):
                qml.CZ(wires=[q, q + 1])

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(data):
            self._ansatz(data, self.variational_params)
            # Expectation of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        self.circuit = circuit

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return raw expectation values as logits."""
        return self.circuit(data)

    def weight_sizes(self) -> List[int]:
        """Return flattened parameter counts."""
        return [self.variational_params.size]

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumClassifier,
                                                                   Iterable[int],
                                                                   List[int],
                                                                   List[qml.PauliOp]]:
    """
    Build the quantum classifier and expose metadata analogous to the ML helper.
    ``encoding`` lists the wires; ``observables`` are the Z‑operators on each qubit.
    """
    model = QuantumClassifier(num_qubits, depth)
    encoding = list(range(num_qubits))
    weight_sizes = model.weight_sizes()
    observables = [qml.PauliZ(i) for i in range(num_qubits)]
    return model, encoding, weight_sizes, observables

__all__ = ["QuantumClassifier", "build_classifier_circuit"]
