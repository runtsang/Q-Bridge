"""Quantum estimator using a variational circuit with entanglement.

The implementation uses Pennylane to build a QNode that
maps 2‑dimensional classical inputs to a single expectation value.
It supports parameter‑shifting for gradient estimation and
provides a simple predict interface compatible with scikit‑learn."""
from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane import qnode
from pennylane import Device
from pennylane.templates import Entanglement
from pennylane.templates import Rot
from pennylane import qchem

# Device: 2 qubits, 100 shots for stochastic gradients
dev = qml.device("default.qubit", wires=2, shots=100)


def _variational_circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Parameterized circuit for 2‑qubit QNN."""
    # Encode inputs via amplitude embedding
    qml.AmplitudeEmbedding(
        features=inputs,
        wires=[0, 1],
        normalize=True,
    )
    # Parameterised rotations
    Rot(weights[0], weights[1], weights[2], wires=0)
    Rot(weights[3], weights[4], weights[5], wires=1)
    # Entangle
    Entanglement(wires=[0, 1], mode="full")
    # Final rotations
    Rot(weights[6], weights[7], weights[8], wires=0)
    Rot(weights[9], weights[10], weights[11], wires=1)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="autograd")
def qnn(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _variational_circuit(inputs, weights)


class EstimatorQNN:
    """Wrapper around the Pennylane QNode to provide a scikit‑learn‑style API."""

    def __init__(self, n_qubits: int = 2, n_weights: int = 12) -> None:
        self.n_qubits = n_qubits
        self.n_weights = n_weights
        # initialise weights randomly
        self.weights = pnp.random.randn(self.n_weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a prediction vector."""
        preds = []
        for x in X:
            preds.append(qnn(x, self.weights))
        return np.array(preds)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.01) -> None:
        """Simple gradient‑descent training loop."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            self.weights = opt.step(lambda w: self._loss(w, X, y), self.weights)

    def _loss(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return np.mean((preds - y) ** 2)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)


__all__ = ["EstimatorQNN"]
