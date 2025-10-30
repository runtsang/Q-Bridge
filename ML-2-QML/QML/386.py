"""Quantum QCNN implementation using PennyLane."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import qnode
from pennylane.templates import StronglyEntanglingLayers
from typing import Tuple


class QCNNModel:
    """
    A variational QCNN built with PennyLane.

    The circuit follows the same logical layers as the classical version:
        - Feature map: ZFeatureMap on 8 qubits
        - Convolutional layers: 3 blocks of 2‑qubit parametrised gates
        - Pooling layers: 3 blocks of 2‑qubit parametrised gates
        - Observable: Z on the first qubit (binary classification)

    The class exposes:
        - `__call__(x)` to evaluate the circuit on a feature vector
        - `train` method to optimise the variational parameters with a chosen optimiser
    """

    def __init__(self, dev: qml.Device | None = None, seed: int = 12345):
        self.dev = dev or qml.device("default.qubit", wires=8, shots=None)
        self._rng = np.random.default_rng(seed)
        self.num_params = 3 * (2 + 2 + 1) + 3 * (2 + 2)  # conv/pool params
        self.weights = pnp.random.uniform(0, 2 * np.pi, self.num_params, requires_grad=True)

        @qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Feature map
            qml.Hadamard(wires=range(8))
            for i in range(8):
                qml.RZ(inputs[i] * np.pi, wires=i)

            # Convolutional layers
            idx = 0
            for _ in range(3):  # 3 conv layers
                for q1, q2 in [(0,1),(2,3),(4,5),(6,7)]:
                    qml.RZ(weights[idx], wires=q1); idx += 1
                    qml.RY(weights[idx], wires=q2); idx += 1
                    qml.CNOT(q1, q2); idx += 1
                    qml.RZ(weights[idx], wires=q1); idx += 1
                # Pooling within conv block
                for q1, q2 in [(0,2),(1,3),(4,6),(5,7)]:
                    qml.RZ(weights[idx], wires=q1); idx += 1
                    qml.RY(weights[idx], wires=q2); idx += 1
                    qml.CNOT(q1, q2); idx += 1

            # Final measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def __call__(self, inputs: np.ndarray) -> float:
        """
        Evaluate the QCNN on a single input vector.
        """
        return float(self.circuit(inputs, self.weights))

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 200,
        loss_fn: str = "mse",
    ) -> Tuple[np.ndarray, float]:
        """
        Simple gradient‑based training loop.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, 8).
        y : np.ndarray
            Binary labels of shape (n_samples,).
        lr : float
            Learning rate.
        epochs : int
            Number of optimisation steps.
        loss_fn : str
            Loss function:'mse' or 'cross_entropy'.

        Returns
        -------
        weights : np.ndarray
            Optimised parameters.
        loss : float
            Final loss value.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)

        for epoch in range(epochs):
            def loss_fn_wrapper(weights):
                preds = np.array([self.circuit(x, weights) for x in X])
                if loss_fn == "mse":
                    return np.mean((preds - y) ** 2)
                else:
                    # Binary cross‑entropy
                    eps = 1e-12
                    preds_clipped = np.clip(preds, eps, 1 - eps)
                    return -np.mean(
                        y * np.log(preds_clipped) + (1 - y) * np.log(1 - preds_clipped)
                    )

            self.weights = opt.step(loss_fn_wrapper, self.weights)

        final_loss = loss_fn_wrapper(self.weights)
        return self.weights, float(final_loss)


def QCNN() -> QCNNModel:
    """
    Factory returning a fully configured QCNNModel instance.
    """
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
