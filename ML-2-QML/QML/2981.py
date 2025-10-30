"""Quantum implementation of a hybrid classifier with a variational ansatz."""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Iterable, Tuple, List

def _quantum_kernel_circuit(x: np.ndarray, wires: List[int]) -> None:
    """Apply a simple two‑qubit quantum kernel to a 2x2 image patch."""
    # Encode each pixel with an Ry rotation
    for idx, wire in enumerate(wires):
        qml.RY(x[idx], wires=wire)
    # Random entangling layer
    for i in range(0, len(wires)-1, 2):
        qml.CNOT(wires=[wires[i], wires[i+1]])
    # Optional second layer
    for i in range(1, len(wires)-1, 2):
        qml.CNOT(wires=[wires[i], wires[i+1]])

def build_classifier_circuit(num_qubits: int, depth: int,
                             device: qml.Device = None) -> Tuple[qml.QNode, Iterable, Iterable, List[qml.operation.Operation]]:
    """
    Construct a variational circuit that encodes 2‑qubit patches and performs classification.
    Returns a QNode, list of encoding parameters, list of variational parameters, and observables.
    """
    if device is None:
        device = qml.device("default.qubit", wires=num_qubits, shots=None)

    # Parameter vectors
    encoding = qml.numpy.array([0.0] * num_qubits)
    weights = qml.numpy.array([0.0] * (num_qubits * depth))

    @qml.qnode(device, interface="autograd")
    def circuit(inputs: np.ndarray, w: np.ndarray) -> np.ndarray:
        # Encode
        for i in range(num_qubits):
            qml.RY(inputs[i], wires=i)
        # Variational layers
        idx = 0
        for _ in range(depth):
            for i in range(num_qubits):
                qml.RY(w[idx], wires=i)
                idx += 1
            for i in range(num_qubits - 1):
                qml.CZ(wires=[i, i+1])
        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    observables = [qml.PauliZ(i) for i in range(num_qubits)]
    return circuit, encoding, weights, observables

class HybridQuantumClassifier:
    """Wrapper around the variational circuit for classification tasks."""
    def __init__(self, num_qubits: int = 4, depth: int = 2):
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        # Initialize parameters
        self.params = np.random.randn(num_qubits * depth)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.circuit(x, self.params)

__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]
