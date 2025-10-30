"""Quantum‑only estimator that can be used as a drop‑in replacement for the
classical EstimatorQNN.  The circuit is a 2‑qubit variational ansatz
parameterised by input and weight parameters.  An expectation value of
PauliZ on each qubit is returned as the feature vector.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Iterable, Sequence, Dict

class EstimatorQNNHybrid:
    """Quantum estimator with the same API as the classical hybrid."""

    def __init__(self, n_qubits: int = 2, n_layers: int = 1) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Parameters
        self.input_params = np.random.randn(n_qubits)
        self.weight_params = np.random.randn(n_layers * n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray) -> np.ndarray:
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(self.weight_params[l * n_qubits + i], wires=i)
                if n_qubits > 1:
                    qml.CNOT([0, 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Return the quantum feature vector for a single input."""
        return self.circuit(inputs)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.evaluate(inputs)


def create_estimator_qnn_hybrid(n_qubits: int = 2, n_layers: int = 1) -> EstimatorQNNHybrid:
    """Factory returning a quantum estimator."""
    return EstimatorQNNHybrid(n_qubits, n_layers)


__all__ = ["EstimatorQNNHybrid", "create_estimator_qnn_hybrid"]
