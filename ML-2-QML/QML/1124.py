from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer

class EstimatorQNN:
    """
    Variational quantum neural network for regression.

    The network encodes a 2‑dimensional input vector via Ry rotations,
    applies a trainable rotation layer with entanglement, and measures
    the expectation value of a Pauli‑Y observable on a single qubit.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: int = 2,
        qubits: int | None = None,
        dev_name: str = "default.qubit",
        seed: int | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.qubits = qubits or input_dim
        self.dev = qml.device(dev_name, wires=self.qubits, shots=1024, seed=seed)

        # Trainable weights: shape (hidden_layers, qubits, 3) for RY,RZ,RX
        self.weights = pnp.random.randn(self.hidden_layers, self.qubits, 3)

        # Observable to measure
        self.obs = qml.PauliY(0)

        # Build the qnode
        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            # Input encoding
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)
            # Variational layers
            for layer in range(self.hidden_layers):
                for q in range(self.qubits):
                    qml.RY(weights[layer, q, 0], wires=q)
                    qml.RZ(weights[layer, q, 1], wires=q)
                    qml.RX(weights[layer, q, 2], wires=q)
                # Entanglement
                for q in range(self.qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[self.qubits - 1, 0])
            return qml.expval(self.obs)

        self.circuit = circuit

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for a batch of inputs."""
        return np.array([self.circuit(x, self.weights) for x in X])

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return np.mean((preds - y) ** 2)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
    ) -> None:
        """Train the quantum circuit using a classical optimizer."""
        opt = AdamOptimizer(lr)
        for epoch in range(epochs):
            self.weights, loss = opt.step_and_cost(lambda w: self.loss(X, y), self.weights)
            if epoch % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} loss: {loss:.4f}")

__all__ = ["EstimatorQNN"]
