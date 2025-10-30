"""Quantum convolution‑pooling network implemented with Pennylane."""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
from typing import Tuple, List


class QCNNHybridQML:
    """
    Variational QCNN that mirrors the classical architecture:
    feature map → convolutional layers → pooling layers → measurement.
    Provides a training loop using automatic differentiation and
    a simple evaluation routine.
    """

    def __init__(self, num_qubits: int = 8, device_name: str = "default.qubit") -> None:
        self.num_qubits = num_qubits
        self.device = qml.device(device_name, wires=num_qubits)
        self.params = self._initialize_params()

        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs, params):
            # Feature map
            for i, val in enumerate(inputs):
                qml.RZ(val, i)

            # Convolutional & pooling layers
            idx = 0
            # First conv layer
            for q in range(0, num_qubits, 2):
                self._conv_block(q, q + 1, params[idx : idx + 3])
                idx += 3
            # First pool layer
            for q in range(0, num_qubits, 2):
                self._pool_block(q, q + 1, params[idx : idx + 3])
                idx += 3
            # Second conv layer (half qubits)
            for q in range(num_qubits // 2, num_qubits, 2):
                self._conv_block(q, q + 1, params[idx : idx + 3])
                idx += 3
            # Second pool layer
            for q in range(num_qubits // 2, num_qubits, 2):
                self._pool_block(q, q + 1, params[idx : idx + 3])
                idx += 3
            # Third conv layer (last two qubits)
            self._conv_block(num_qubits - 2, num_qubits - 1, params[idx : idx + 3])
            idx += 3
            # Third pool layer
            self._pool_block(num_qubits - 2, num_qubits - 1, params[idx : idx + 3])

            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def _conv_block(self, q1: int, q2: int, params: List[float]) -> None:
        """Two‑qubit convolution block."""
        qml.RZ(-np.pi / 2, q2)
        qml.CNOT(q2, q1)
        qml.RZ(params[0], q1)
        qml.RY(params[1], q2)
        qml.CNOT(q1, q2)
        qml.RY(params[2], q2)
        qml.CNOT(q2, q1)
        qml.RZ(np.pi / 2, q1)

    def _pool_block(self, q1: int, q2: int, params: List[float]) -> None:
        """Two‑qubit pooling block."""
        qml.RZ(-np.pi / 2, q2)
        qml.CNOT(q2, q1)
        qml.RZ(params[0], q1)
        qml.RY(params[1], q2)
        qml.CNOT(q1, q2)
        qml.RY(params[2], q2)

    def _initialize_params(self) -> np.ndarray:
        """Randomly initialise all variational parameters."""
        n_params = 3 * (self.num_qubits // 2 + self.num_qubits // 4 + 1)
        return pnp.random.uniform(0, 2 * np.pi, size=n_params)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass returning a scalar in [0, 1] via sigmoid.

        Parameters
        ----------
        inputs : array-like
            Shape (batch, num_qubits). Values assumed to be in [-π, π].

        Returns
        -------
        np.ndarray
            Predictions in [0, 1].
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        preds = np.array([self.circuit(x, self.params) for x in inputs])
        return 1 / (1 + np.exp(-preds))

    def loss(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Binary cross‑entropy loss."""
        preds = self.predict(inputs)
        return -np.mean(targets * np.log(preds + 1e-12) + (1 - targets) * np.log(1 - preds + 1e-12))

    def train(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        """Full training loop using gradient descent."""
        for epoch in range(epochs):
            loss_val, grads = qml.grad(self.loss, argnum=0)(inputs, targets)
            self.params -= lr * grads
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs} – Loss: {loss_val:.4f}")

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Accuracy on a binary classification task."""
        preds = (self.predict(inputs) > 0.5).astype(int)
        return np.mean(preds == targets)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Convenience wrapper for prediction."""
        return self.predict(inputs)


__all__ = ["QCNNHybridQML"]
