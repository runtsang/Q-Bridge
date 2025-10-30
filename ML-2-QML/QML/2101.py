"""Hybrid quantum circuit using Pennylane with a Z‑feature map and modular entangling blocks."""

import pennylane as qml
import pennylane.numpy as pnp
from pennylane import numpy as np
from pennylane.measurements import Probability
from pennylane.optimize import AdamOptimizer
from typing import Callable, Tuple

__all__ = ["QCNN", "QuantumCNN"]


def _entangling_block(qubits: list[int], params: pnp.ndarray) -> None:
    """Entangling block for a given set of qubits."""
    for i in range(len(qubits) - 1):
        qml.CNOT(wires=[qubits[i], qubits[i + 1]])
    # Apply a parameterized rotation
    for w, p in zip(qubits, params):
        qml.RY(p, wires=w)


def _feature_map(num_qubits: int, data: pnp.ndarray) -> None:
    """Z‑feature map: encode data as rotations around Z."""
    for i in range(num_qubits):
        qml.RZ(data[i], wires=i)


class QuantumCNN:
    """A variational quantum circuit mimicking a QCNN."""

    def __init__(self, num_qubits: int = 8, depth: int = 3, seed: int = 42):
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=num_qubits)
        # Parameter shapes
        self.feature_shape = (num_qubits,)
        self.weight_shape = (depth, num_qubits)
        self.params = pnp.random.RandomState(seed).randn(*self.weight_shape)

    @qml.qnode
    def circuit(self, data: pnp.ndarray, weights: pnp.ndarray) -> pnp.ndarray:
        _feature_map(self.num_qubits, data)
        for layer in range(self.depth):
            _entangling_block(list(range(self.num_qubits)), weights[layer])
        return qml.expval(qml.PauliZ(0))

    def predict(self, data: pnp.ndarray) -> pnp.ndarray:
        """Return a probability (sigmoid of expectation)."""
        exp_val = self.circuit(data, self.params)
        return 1 / (1 + pnp.exp(-exp_val))

    def loss(self, data: pnp.ndarray, labels: pnp.ndarray) -> float:
        preds = self.predict(data)
        return - (labels * pnp.log(preds + 1e-12) + (1 - labels) * pnp.log(1 - preds + 1e-12)).mean()

    def train(self, data: pnp.ndarray, labels: pnp.ndarray, epochs: int = 200, lr: float = 0.05):
        opt = AdamOptimizer(lr)
        for epoch in range(epochs):
            loss_grad = qml.grad(lambda w: self.loss(data, labels))(self.params)
            self.params = opt.step(loss_grad, self.params)
            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | Loss: {self.loss(data, labels):.4f}")

    def __call__(self, data: pnp.ndarray) -> pnp.ndarray:
        return self.predict(data)


def QCNN() -> QuantumCNN:
    """Factory returning a quantum QCNN instance."""
    return QuantumCNN()
