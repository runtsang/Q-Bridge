"""Hybrid quantum‑classical QCNN implemented with PennyLane.

The network builds a parameterised circuit that mimics the
convolution‑pooling structure of the original design.  The
parameters are optimised with a gradient‑based optimiser
implemented in PennyLane.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QCNNModel:
    """Hybrid QCNN using PennyLane.

    Parameters
    ----------
    num_qubits : int, optional
        Number of qubits in the feature map and ansatz.  Defaults to 8.
    """

    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.feature_map = qml.templates.feature_maps.ZFeatureMap(num_qubits)
        self.num_params = num_qubits * 3 * 4  # 4 convolution layers
        self.params = pnp.random.randn(self.num_params)
        self.qnode = self._build_qnode()

    def _conv_circuit(self, q1: int, q2: int, params: np.ndarray):
        """Convolution sub‑circuit used by all layers."""
        qml.RZ(-np.pi / 2, wires=q2)
        qml.CNOT(wires=[q2, q1])
        qml.RZ(params[0], wires=q1)
        qml.RY(params[1], wires=q2)
        qml.CNOT(wires=[q1, q2])
        qml.RY(params[2], wires=q2)
        qml.CNOT(wires=[q2, q1])
        qml.RZ(np.pi / 2, wires=q1)

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, params):
            # Encode classical data
            self.feature_map(x)
            idx = 0
            # First convolutional layer on 4 pairs
            for q1, q2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
                self._conv_circuit(q1, q2, params[idx : idx + 3])
                idx += 3
            # Second convolutional layer on 2 pairs
            for q1, q2 in [(0, 2), (4, 6)]:
                self._conv_circuit(q1, q2, params[idx : idx + 3])
                idx += 3
            # Third convolutional layer on the remaining pair
            self._conv_circuit(0, 4, params[idx : idx + 3])
            # Expectation value of Pauli‑Z on qubit 0
            return qml.expval(qml.PauliZ(0))
        return circuit

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for a batch of inputs."""
        return np.array([self.qnode(x, self.params) for x in X])

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 200,
        verbose: bool = False,
    ):
        """Gradient‑descent training loop."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)

        def loss_fn(params, x, target):
            return (self.qnode(x, params) - target) ** 2

        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                grads = qml.grad(loss_fn)(self.params, xi, yi)
                self.params = opt.apply_gradients(self.params, grads)
            if verbose and (epoch + 1) % 20 == 0:
                loss = np.mean(
                    [loss_fn(self.params, xi, yi) for xi, yi in zip(X, y)]
                )
                print(f"Epoch {epoch+1}/{epochs} – loss: {loss:.4f}")

    def __repr__(self) -> str:
        return f"<QCNNModel num_qubits={self.num_qubits} params={self.params.shape[0]}>"


def QCNN() -> QCNNModel:
    """Factory that returns a ready‑to‑train :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
