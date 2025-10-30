"""Quantum neural network for regression using Pennylane.

This implementation expands the original single‑qubit example to a
multi‑qubit variational circuit with configurable depth and entanglement.
It demonstrates how classical data can be encoded into quantum gates,
and how the expectation value of a Pauli‑Z observable can be used as the
output of a regression model.
"""

import pennylane as qml
import numpy as np
from typing import Sequence

class EstimatorNN:
    """Hybrid quantum‑classical regressor."""

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        observable: str | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        if observable is None:
            self.observable = "Z" * n_qubits
        else:
            self.observable = observable

        # Random initial weights for each variational layer
        self.weights = np.random.randn(n_layers, n_qubits, 3)

    def circuit(self, inputs: Sequence[float], weights: np.ndarray) -> float:
        """Variational circuit with feature encoding and entangling layers."""
        for i, val in enumerate(inputs):
            qml.RY(val, wires=i)

        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                qml.Rot(
                    weights[layer, qubit, 0],
                    weights[layer, qubit, 1],
                    weights[layer, qubit, 2],
                    wires=qubit,
                )
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        return qml.expval(qml.PauliZ(*range(self.n_qubits)))

    def __call__(self, inputs: Sequence[float]) -> float:
        return self.circuit(inputs, self.weights)

def EstimatorQNN() -> EstimatorNN:
    """Return a default quantum regressor."""
    return EstimatorNN()

__all__ = ["EstimatorQNN"]
