"""Quantum neural network estimator with entangled variational circuit.

The class `EstimatorQNNExtended` implements a 2‑qubit circuit that encodes 2‑D inputs, applies entangling layers,
and measures a Pauli‑Z⊗Z observable.  A simple gradient‑descent optimiser trains the two weight parameters.
"""

import pennylane as qml
import numpy as np

# 2‑qubit device
dev = qml.device("default.qubit", wires=2)


def _circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
    """Variational circuit returning the expectation of Z⊗Z."""
    # Input encoding: RX rotations
    qml.RX(inputs[0], wires=0)
    qml.RX(inputs[1], wires=1)

    # Entangling layer
    qml.CNOT(wires=[0, 1])

    # Parameterised rotation layer
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)

    # Second entangling layer
    qml.CNOT(wires=[1, 0])

    # Measurement
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


# QNode wrapping the circuit
qnode = qml.QNode(_circuit, dev)


class EstimatorQNNExtended:
    """Quantum estimator that maps 2‑D inputs to a scalar output via a variational circuit."""

    def __init__(self, init_weights: np.ndarray | None = None) -> None:
        if init_weights is None:
            self.weights = np.random.randn(2)
        else:
            self.weights = np.array(init_weights, dtype=float)

    def predict(self, inputs: np.ndarray) -> float:
        """Evaluate the circuit for a single input vector."""
        return qnode(inputs, self.weights)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 200,
    ) -> None:
        """Simple batch‑gradient descent training loop."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)

        for _ in range(epochs):
            # Compute mean‑squared‑error loss
            loss = np.mean(
                (qnode(X[i], self.weights) - y[i]) ** 2 for i in range(len(X))
            )
            # Update weights
            self.weights = opt.step(
                lambda w: np.mean(
                    (qnode(X[i], w) - y[i]) ** 2 for i in range(len(X))
                ),
                self.weights,
            )


def EstimatorQNN() -> EstimatorQNNExtended:
    """Return an instance of the quantum estimator."""
    return EstimatorQNNExtended()


__all__ = ["EstimatorQNN", "EstimatorQNNExtended"]
