"""Quantum regression estimator using a variational circuit."""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from typing import Iterable

class EstimatorQNN:
    """
    Variational quantum circuit that maps 2‑dimensional inputs to a single expectation value.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit. Defaults to 1.
    hidden_layers : int
        Number of rotation‑layer repetitions. Defaults to 2.
    """
    def __init__(self, num_qubits: int = 1, hidden_layers: int = 2) -> None:
        self.num_qubits = num_qubits
        self.hidden_layers = hidden_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, x1: float, x2: float, *weights: float) -> float:
        # Encode inputs
        qml.RY(x1, wires=0)
        qml.RZ(x2, wires=0)
        # Variational layers
        idx = 0
        for _ in range(self.hidden_layers):
            for w in weights[idx : idx + self.num_qubits * 3]:
                qml.RY(w, wires=0)
                idx += 1
        # Measurement
        return qml.expval(qml.PauliZ(0))

    def __call__(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Evaluate the circuit for a batch of inputs."""
        return np.array([self.qnode(x1, x2, *weights) for x1, x2 in x])

    def init_weights(self, seed: int = 42) -> np.ndarray:
        """Randomly initialize variational parameters."""
        rng = np.random.default_rng(seed)
        return rng.normal(0, np.pi, self.hidden_layers * self.num_qubits * 3)

def EstimatorQNN_factory(
    num_qubits: int = 1,
    hidden_layers: int = 2,
) -> EstimatorQNN:
    """Convenience factory that returns a ready‑to‑use EstimatorQNN instance."""
    return EstimatorQNN(num_qubits=num_qubits, hidden_layers=hidden_layers)

__all__ = ["EstimatorQNN", "EstimatorQNN_factory"]
