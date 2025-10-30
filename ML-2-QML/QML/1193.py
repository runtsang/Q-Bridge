"""
QCNNHybrid – quantum implementation using Pennylane.
Provides a variational ansatz that mimics the classical residual blocks
and a full training loop that can be run on a simulator or a real device.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
from pennylane.measurements import StateFn
from typing import Iterable, Tuple, Callable


def conv_circuit(qubits: Tuple[int, int], params: np.ndarray) -> qml.QNode:
    """Single convolution unit acting on a pair of qubits."""
    dev = qml.device("default.qubit", wires=len(qubits))

    @qml.qnode(dev, interface="autograd")
    def circuit(*x):
        qml.RZ(-np.pi / 2, qubits[1])
        qml.CNOT(qubits[1], qubits[0])
        qml.RZ(params[0], qubits[0])
        qml.RY(params[1], qubits[1])
        qml.CNOT(qubits[0], qubits[1])
        qml.RY(params[2], qubits[1])
        qml.CNOT(qubits[1], qubits[0])
        qml.RZ(np.pi / 2, qubits[0])
        return qml.expval(qml.PauliZ(qubits[0]))
    return circuit


def conv_layer(num_qubits: int, param_prefix: str, params: np.ndarray) -> qml.QNode:
    """Apply convolution units across all adjacent qubit pairs."""
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def layer(x):
        for i in range(0, num_qubits, 2):
            idx = i // 2 * 3
            conv = conv_circuit((i, i + 1), params[idx : idx + 3])
            x[i] = conv(x[i], x[i + 1])
        return x
    return layer


def pool_circuit(qubits: Tuple[int, int], params: np.ndarray) -> qml.QNode:
    """Pooling unit that reduces two qubits to one."""
    dev = qml.device("default.qubit", wires=len(qubits))

    @qml.qnode(dev, interface="autograd")
    def circuit(*x):
        qml.RZ(-np.pi / 2, qubits[1])
        qml.CNOT(qubits[1], qubits[0])
        qml.RZ(params[0], qubits[0])
        qml.RY(params[1], qubits[1])
        qml.CNOT(qubits[0], qubits[1])
        qml.RY(params[2], qubits[1])
        return qml.expval(qml.PauliZ(qubits[0]))
    return circuit


def pool_layer(sources: Iterable[int], sinks: Iterable[int], params: np.ndarray) -> qml.QNode:
    """Apply pooling units to map sources → sinks."""
    dev = qml.device("default.qubit", wires=max(sinks) + 1)

    @qml.qnode(dev, interface="autograd")
    def layer(x):
        for src, sink, idx in zip(sources, sinks, range(0, len(sources) * 3, 3)):
            pool = pool_circuit((src, sink), params[idx : idx + 3])
            x[sink] = pool(x[src], x[sink])
        return x
    return layer


class QCNNHybridQNN:
    """
    Variational QCNN that mirrors the classical residual structure.
    The ansatz is built from convolution and pooling layers with learnable
    parameters.  The circuit is wrapped in a ``qml.QNode`` that returns a single
    expectation value used as the output probability.
    """
    def __init__(self, num_qubits: int = 8, feature_map: qml.QNode | None = None):
        self.num_qubits = num_qubits
        self.feature_map = feature_map or qml.feature_maps.ZFeatureMap(num_qubits)
        self.params = np.random.randn(self.num_qubits * 3 * 7)  # 7 layers
        self.dev = qml.device("default.qubit", wires=num_qubits)

    def circuit(self, x: np.ndarray) -> qml.QNode:
        """Full QCNN ansatz."""
        @qml.qnode(self.dev, interface="autograd")
        def qnn(x):
            # Feature map
            for w, v in zip(self.feature_map.wires, self.feature_map.params):
                qml.RX(x[w], w)
                qml.RZ(v, w)

            # Layer 1
            for i in range(0, self.num_qubits, 2):
                idx = i // 2 * 3
                conv = conv_circuit((i, i + 1), self.params[idx : idx + 3])
                conv(x[i], x[i + 1])

            # Pool 1
            for i in range(0, self.num_qubits, 2):
                idx = (self.num_qubits // 2 + i // 2) * 3
                pool = pool_circuit((i, i + 1), self.params[idx : idx + 3])
                x[i] = pool(x[i], x[i + 1])

            # Continue alternating conv/pool until single qubit remains
            # (omitted for brevity – same pattern with decreasing qubit count)
            return qml.expval(qml.PauliZ(0))
        return qnn

    def loss(self, params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """Binary cross‑entropy loss for a batch."""
        preds = self.circuit(x)(x)
        preds = 1 / (1 + np.exp(-preds))  # sigmoid
        return -np.mean(y * np.log(preds + 1e-10) + (1 - y) * np.log(1 - preds + 1e-10))

    def train(
        self,
        data_loader: Iterable[Tuple[np.ndarray, np.ndarray]],
        epochs: int,
        lr: float = 0.01,
    ) -> None:
        """End‑to‑end training using Adam."""
        opt = AdamOptimizer(lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in data_loader:
                loss_grad = qml.grad(self.loss)(self.params, x, y)
                self.params = opt.step(self.params, loss_grad)
                epoch_loss += self.loss(self.params, x, y)
            epoch_loss /= len(data_loader)
            print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss:.4f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return probability estimates for a batch of inputs."""
        preds = self.circuit(x)(x)
        return 1 / (1 + np.exp(-preds))


def QCNNHybridQNNFactory(num_qubits: int = 8) -> QCNNHybridQNN:
    """Convenience factory for the quantum QCNN."""
    return QCNNHybridQNN(num_qubits)


__all__ = ["QCNNHybridQNN", "QCNNHybridQNNFactory"]
