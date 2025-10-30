"""Quantum convolutional neural network implemented with Pennylane.

The architecture mirrors the original Qiskit QCNN.  It uses a
parameter‑shift gradient for training and a simple stochastic
gradient descent loop.  The circuit is fully differentiable and
runs on the local default.qubit simulator or a GPU device.
"""

import pennylane as qml
import pennylane.numpy as pnp
from pennylane.optimize import AdamOptimizer
import numpy as np


class QCNNGen121:
    """Quantum convolutional neural network with a Pennylane QNode."""

    def __init__(self, n_qubits: int = 8, seed: int = 42) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        np.random.seed(seed)
        # 42 parameters: 12 (conv1) + 12 (pool1) + 6 (conv2) + 6 (pool2) + 3 (conv3) + 3 (pool3)
        self.params = pnp.random.uniform(0, 2 * np.pi, 42, requires_grad=True)
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray) -> np.ndarray:
            # Feature map
            for i, w in enumerate(inputs):
                qml.PhaseShift(w, wires=i)

            idx = 0
            # Conv1: 4 pairs
            for i in range(0, 8, 2):
                qml.RZ(self.params[idx], wires=i)
                qml.RY(self.params[idx + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(self.params[idx + 2], wires=i)
                idx += 3

            # Pool1: 4 pairs
            for i in range(0, 8, 2):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(self.params[idx], wires=i)
                qml.RY(self.params[idx + 1], wires=i + 1)
                qml.CNOT(wires=[i + 1, i])
                idx += 3

            # Conv2: 2 pairs
            for i in range(0, 4, 2):
                qml.RZ(self.params[idx], wires=i)
                qml.RY(self.params[idx + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(self.params[idx + 2], wires=i)
                idx += 3

            # Pool2: 2 pairs
            for i in range(0, 4, 2):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(self.params[idx], wires=i)
                qml.RY(self.params[idx + 1], wires=i + 1)
                qml.CNOT(wires=[i + 1, i])
                idx += 3

            # Conv3: 1 pair
            for i in range(0, 2, 2):
                qml.RZ(self.params[idx], wires=i)
                qml.RY(self.params[idx + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(self.params[idx + 2], wires=i)
                idx += 3

            # Pool3: 1 pair
            for i in range(0, 2, 2):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(self.params[idx], wires=i)
                qml.RY(self.params[idx + 1], wires=i + 1)
                qml.CNOT(wires=[i + 1, i])
                idx += 3

            # Output expectation value of Z on qubit 0
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.circuit(inputs)

    def loss(self, inputs: np.ndarray, target: float) -> float:
        """Mean‑squared‑error loss."""
        out = self.__call__(inputs)
        return (out - target) ** 2

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
        batch_size: int = 32,
    ) -> None:
        """Simple training loop using Adam."""
        opt = AdamOptimizer(lr)
        n_samples = X.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                grads = opt.grad(lambda p: np.mean([self.loss(x, t) for x, t in zip(X_batch, y_batch)]), self.params)
                self.params = opt.step(grads, self.params)

__all__ = ["QCNNGen121"]
