"""Quantum QCNN using PennyLane with a hybrid variational circuit and training utilities."""

import pennylane as qml
import pennylane.numpy as np
from pennylane.optimize import AdamOptimizer
from typing import List

class QCNNModel:
    """Quantum QCNN model built with PennyLane, featuring convolution and pooling layers."""

    def __init__(self, n_qubits: int = 8, seed: int = 123) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.seed = seed
        np.random.seed(seed)
        self.feature_map = qml.templates.AngleEmbedding
        self.build_circuit()
        self.qnode = qml.QNode(self.circuit, self.dev, interface="autograd")

    def conv_layer(self, params: np.ndarray, qubits: List[int]) -> None:
        """Apply a 2‑qubit convolution block to the given qubits."""
        for i in range(0, len(qubits), 2):
            qml.RZ(-np.pi / 2, wires=qubits[i + 1])
            qml.CNOT(wires=[qubits[i + 1], qubits[i]])
            qml.RZ(params[i], wires=qubits[i])
            qml.RY(params[i + 1], wires=qubits[i + 1])
            qml.CNOT(wires=[qubits[i], qubits[i + 1]])
            qml.RY(params[i + 2], wires=qubits[i + 1])
            qml.CNOT(wires=[qubits[i + 1], qubits[i]])
            qml.RZ(np.pi / 2, wires=qubits[i])

    def pool_layer(self, params: np.ndarray, source: int, sink: int) -> None:
        """Apply a 2‑qubit pooling block between source and sink qubits."""
        qml.RZ(-np.pi / 2, wires=sink)
        qml.CNOT(wires=[sink, source])
        qml.RZ(params[0], wires=source)
        qml.RY(params[1], wires=sink)
        qml.CNOT(wires=[source, sink])
        qml.RY(params[2], wires=sink)

    def build_circuit(self) -> None:
        """Construct the full QCNN circuit with feature map, convolution, pooling, and readout."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Feature map
            self.feature_map(inputs, wires=range(self.n_qubits))
            # Convolution and pooling layers
            idx = 0
            # First conv layer (8 qubits)
            self.conv_layer(weights[idx : idx + 3 * (self.n_qubits // 2)], range(self.n_qubits))
            idx += 3 * (self.n_qubits // 2)
            # First pool layer (4 qubits)
            for src, sink in [(0, 4), (1, 5), (2, 6), (3, 7)]:
                self.pool_layer(weights[idx : idx + 3], src, sink)
                idx += 3
            # Second conv layer (4 qubits)
            self.conv_layer(weights[idx : idx + 3 * 2], range(4, 8))
            idx += 3 * 2
            # Second pool layer (2 qubits)
            for src, sink in [(0, 4), (1, 5)]:
                self.pool_layer(weights[idx : idx + 3], src, sink)
                idx += 3
            # Third conv layer (2 qubits)
            self.conv_layer(weights[idx : idx + 3], range(6, 8))
            idx += 3
            # Third pool layer (1 qubit)
            self.pool_layer(weights[idx : idx + 3], 0, 1)
            idx += 3
            # Readout: expectation of Z on the last qubit
            return qml.expval(qml.PauliZ(7))

        self.circuit = circuit

    def loss(self, params: np.ndarray, data: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Binary cross‑entropy loss over a batch."""
        preds = np.array([self.circuit(x, params) for x in data])
        preds = 1 / (1 + np.exp(-preds))  # sigmoid
        return -np.mean(targets * np.log(preds + 1e-10) + (1 - targets) * np.log(1 - preds + 1e-10))

    def train(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> np.ndarray:
        """Simple gradient‑based training loop using Adam."""
        n_weights = self.circuit.num_params
        weights = np.random.randn(n_weights)
        opt = AdamOptimizer(lr)
        for epoch in range(epochs):
            weights, loss_val = opt.step_and_cost(lambda w: self.loss(w, data, targets), weights)
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:04d} | Loss: {loss_val:.6f}")
        return weights

    def predict(self, data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Predict binary labels for new data."""
        probs = np.array([self.circuit(x, weights) for x in data])
        probs = 1 / (1 + np.exp(-probs))
        return (probs > 0.5).astype(int)


def QCNN() -> QCNNModel:
    """Factory returning a configured QCNNModel."""
    return QCNNModel()


__all__ = ["QCNNModel", "QCNN"]
