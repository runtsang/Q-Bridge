"""Variational QCNN implemented in Pennylane with explicit pooling."""

import pennylane as qml
import pennylane.numpy as np
from pennylane import qnn
from pennylane.optimize import AdamOptimizer
from pennylane.templates import StronglyEntanglingLayers
from typing import Tuple


class QCNNQuantum:
    """Quantum convolutional neural network.

    The circuit consists of an angle‑embedding feature map followed by a stack of
    convolutional layers implemented with ``StronglyEntanglingLayers``.  After
    each convolution a qubit is measured and reset to realise a simple pooling
    operation.  The output is the expectation value of ``PauliZ`` on the first
    qubit.

    Parameters
    ----------
    n_qubits : int, default 8
        Number of qubits in the device.
    conv_layers : int, default 3
        Number of convolutional (and pooling) layers.
    """

    def __init__(self, n_qubits: int = 8, conv_layers: int = 3) -> None:
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.n_qubits = n_qubits
        self.conv_layers = conv_layers
        self.circuit = self._build_circuit()
        self.qnn = qnn.QNode(self.circuit, self.dev, interface="autograd")

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Feature embedding
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
            # Convolution + pooling
            for l in range(self.conv_layers):
                StronglyEntanglingLayers(weights[l], wires=range(self.n_qubits))
                # Pooling: measure qubit 0 and reset it
                qml.measure(0)
                qml.reset(0)
            return qml.expval(qml.PauliZ(0))
        return circuit

    def cost(self, params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        preds = self.qnn(x, params)
        return np.mean((preds - y) ** 2)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
    ) -> np.ndarray:
        opt = AdamOptimizer(lr)
        # Initialise weights: shape (conv_layers, n_qubits, 3)
        params = np.random.randn(self.conv_layers, self.n_qubits, 3)
        for _ in range(epochs):
            params, _ = opt.step_and_cost(
                lambda p: self.cost(p, x_train, y_train), params
            )
        return params


def QCNN() -> QCNNQuantum:
    """Return a QCNNQuantum instance with default settings."""
    return QCNNQuantum()


__all__ = ["QCNNQuantum", "QCNN"]
