"""QCNNModel: quantum convolution‑inspired network using Pennylane."""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Iterable, Sequence


def _conv_circuit(wires: Sequence[int], params: Sequence[float]) -> None:
    """Two‑qubit convolution block."""
    qml.RZ(-np.pi / 2, wires[wires[1]])
    qml.CNOT(wires[wires[1]], wires[wires[0]])
    qml.RZ(params[0], wires[wires[0]])
    qml.RY(params[1], wires[wires[1]])
    qml.CNOT(wires[wires[0]], wires[wires[1]])
    qml.RY(params[2], wires[wires[1]])
    qml.CNOT(wires[wires[1]], wires[wires[0]])
    qml.RZ(np.pi / 2, wires[wires[0]])


def _pool_circuit(wires: Sequence[int], params: Sequence[float]) -> None:
    """Two‑qubit pooling block."""
    qml.RZ(-np.pi / 2, wires[wires[1]])
    qml.CNOT(wires[wires[1]], wires[wires[0]])
    qml.RZ(params[0], wires[wires[0]])
    qml.RY(params[1], wires[wires[1]])
    qml.CNOT(wires[wires[0]], wires[wires[1]])
    qml.RY(params[2], wires[wires[1]])


class QCNNModel:
    """
    Quantum QCNN implemented with Pennylane.

    The circuit consists of a feature map followed by three convolutional layers
    and three pooling layers, mirroring the classical net.
    Parameters are grouped into weight tensors for each block; all are
    differentiable via the parameter‑shift rule.
    """

    def __init__(self, dev: qml.Device | None = None, seed: int | None = None) -> None:
        self.dev = dev or qml.device("default.qubit", wires=8)
        if seed is not None:
            np.random.seed(seed)

        # Parameter initialization
        self.weight_params = {
            "c1": np.random.randn(8 // 2 * 3),
            "p1": np.random.randn(8 // 2 * 3),
            "c2": np.random.randn(4 // 2 * 3),
            "p2": np.random.randn(4 // 2 * 3),
            "c3": np.random.randn(2 // 2 * 3),
            "p3": np.random.randn(2 // 2 * 3),
        }

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: dict[str, np.ndarray]) -> float:
            # Feature‑map encoding
            qml.templates.feature_maps.ZFeatureMap(wires=8)(inputs)

            # First convolution & pooling
            for i in range(0, 8, 2):
                _conv_circuit([i, i + 1], weights["c1"][i // 2 * 3 : i // 2 * 3 + 3])
            for i in range(0, 8, 2):
                _pool_circuit([i, i + 1], weights["p1"][i // 2 * 3 : i // 2 * 3 + 3])

            # Second convolution & pooling
            for i in range(4, 8, 2):
                _conv_circuit([i, i + 1], weights["c2"][(i - 4) // 2 * 3 : (i - 4) // 2 * 3 + 3])
            for i in range(4, 8, 2):
                _pool_circuit([i, i + 1], weights["p2"][(i - 4) // 2 * 3 : (i - 4) // 2 * 3 + 3])

            # Third convolution & pooling
            for i in range(6, 8, 2):
                _conv_circuit([i, i + 1], weights["c3"][(i - 6) // 2 * 3 : (i - 6) // 2 * 3 + 3])
            for i in range(6, 8, 2):
                _pool_circuit([i, i + 1], weights["p3"][(i - 6) // 2 * 3 : (i - 6) // 2 * 3 + 3])

            # Measurement on first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def __call__(self, inputs: np.ndarray) -> float:
        """Return the sigmoid‑mapped expectation value."""
        raw = self.circuit(inputs, self.weight_params)
        return 1 / (1 + np.exp(-raw))

    def parameters(self) -> dict[str, np.ndarray]:
        """Return the current weight parameters."""
        return self.weight_params

    def set_parameters(self, new_params: dict[str, np.ndarray]) -> None:
        """Replace the current parameters."""
        self.weight_params = new_params

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> None:
        """Simple supervised training loop using gradient descent."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)

        loss_fn = lambda out, target: (out - target) ** 2

        for epoch in range(1, epochs + 1):
            for x, t in zip(X, y):
                params, _ = opt.step_and_cost(
                    lambda p: loss_fn(self.circuit(x, p), t),
                    self.weight_params,
                )
                self.weight_params = params

            if verbose and epoch % 10 == 0:
                preds = np.array([self(x) for x in X])
                loss = np.mean((preds - y) ** 2)
                print(f"Epoch {epoch:3d} | loss: {loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorised prediction."""
        return np.array([self(x) for x in X])


def QCNN() -> QCNNModel:
    """Convenience factory returning a pre‑configured QCNNModel."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
