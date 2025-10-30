"""Quantum convolutional neural network implemented with PennyLane."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer


class QCNNModel:
    """Variational QCNN with parameter‑shift gradients and optional noise.

    The circuit architecture follows the original QCNN design but is expressed
    using PennyLane’s higher‑level primitives.  A feature map, convolutional
    layers and pooling layers are all built from reusable sub‑circuits.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        shots: int = 8192,
        noise: bool = False,
        device_name: str = "default.qubit",
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.noise = noise

        if noise:
            # Simple depolarising noise model for demonstration
            noise_model = qml.transforms.depolarizing_noise(0.01)
            self.dev = qml.device(device_name, wires=n_qubits, shots=shots, noise=noise_model)
        else:
            self.dev = qml.device(device_name, wires=n_qubits, shots=shots)

        self.params = None  # will be initialised in build_ansatz

    # ------------------------------------------------------------------
    # Feature map
    # ------------------------------------------------------------------
    def _feature_map(self, x: np.ndarray) -> qml.QNode:
        """Z‑feature map from Qiskit, implemented in PennyLane."""
        def circuit(x):
            for i, val in enumerate(x):
                qml.RZ(val, wires=i)
            return qml.expval(qml.PauliZ(0))
        return qml.QNode(circuit, self.dev, interface="autograd")

    # ------------------------------------------------------------------
    # Convolution and pooling primitives
    # ------------------------------------------------------------------
    def _conv_circuit(self, params: np.ndarray, wires: tuple[int,...]) -> None:
        """Two‑qubit convolution unit."""
        qml.RZ(-np.pi / 2, wires=wires[1])
        qml.CNOT(wires=wires[1], control=wires[0])
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires=wires[0], control=wires[1])
        qml.RY(params[2], wires=wires[1])
        qml.CNOT(wires=wires[1], control=wires[0])
        qml.RZ(np.pi / 2, wires=wires[0])

    def _pool_circuit(self, params: np.ndarray, wires: tuple[int,...]) -> None:
        """Two‑qubit pooling unit."""
        qml.RZ(-np.pi / 2, wires=wires[1])
        qml.CNOT(wires=wires[1], control=wires[0])
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires=wires[0], control=wires[1])
        qml.RY(params[2], wires=wires[1])

    # ------------------------------------------------------------------
    # Build the full ansatz
    # ------------------------------------------------------------------
    def build_ansatz(self) -> None:
        """Constructs the full QCNN ansatz and stores the parameter shape."""
        n = self.n_qubits
        # Layer 1: 4 qubits → 2 conv + 2 pool
        conv1_params = pnp.random.randn(n // 2, 3)
        pool1_params = pnp.random.randn(n // 2, 3)
        # Layer 2: 4 qubits → 2 conv + 2 pool
        conv2_params = pnp.random.randn(n // 4, 3)
        pool2_params = pnp.random.randn(n // 4, 3)
        # Layer 3: 2 qubits → 1 conv + 1 pool
        conv3_params = pnp.random.randn(n // 8, 3)
        pool3_params = pnp.random.randn(n // 8, 3)

        self.params = {
            "conv1": conv1_params,
            "pool1": pool1_params,
            "conv2": conv2_params,
            "pool2": pool2_params,
            "conv3": conv3_params,
            "pool3": pool3_params,
        }

    # ------------------------------------------------------------------
    # QNode
    # ------------------------------------------------------------------
    def _qnode(self, x: np.ndarray) -> qml.QNode:
        """Return a QNode that evaluates the QCNN on input x."""
        def circuit(x):
            # Feature map
            for i, val in enumerate(x):
                qml.RZ(val, wires=i)

            # Convolutional and pooling layers
            # Layer 1
            for i in range(0, self.n_qubits, 2):
                self._conv_circuit(self.params["conv1"][i // 2], wires=(i, i + 1))
            for i in range(0, self.n_qubits, 2):
                self._pool_circuit(self.params["pool1"][i // 2], wires=(i, i + 1))

            # Layer 2
            for i in range(0, self.n_qubits // 2, 2):
                self._conv_circuit(self.params["conv2"][i // 4], wires=(i + self.n_qubits // 2, i + self.n_qubits // 2 + 1))
            for i in range(0, self.n_qubits // 2, 2):
                self._pool_circuit(self.params["pool2"][i // 4], wires=(i + self.n_qubits // 2, i + self.n_qubits // 2 + 1))

            # Layer 3
            self._conv_circuit(self.params["conv3"][0], wires=(self.n_qubits - 2, self.n_qubits - 1))
            self._pool_circuit(self.params["pool3"][0], wires=(self.n_qubits - 2, self.n_qubits - 1))

            # Measurement
            return qml.expval(qml.PauliZ(0))
        return qml.QNode(circuit, self.dev, interface="autograd")

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> None:
        """Train the QCNN using Adam and the parameter‑shift rule."""
        self.build_ansatz()
        opt = AdamOptimizer(lr)

        for epoch in range(epochs):
            loss = 0.0
            for xi, yi in zip(X, y):
                qnode = self._qnode(xi)
                pred = qnode(xi)
                loss += (pred - yi) ** 2
            loss /= len(X)
            opt.step(self.params, lambda p: self._loss_function(p, X, y))
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {loss:.4f}")

    def _loss_function(self, params, X, y):
        """Compute MSE loss over the dataset."""
        loss = 0.0
        for xi, yi in zip(X, y):
            qnode = self._qnode(xi)
            pred = qnode(xi)
            loss += (pred - yi) ** 2
        return loss / len(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions for a batch of inputs."""
        probs = np.array([self._qnode(xi)(xi) for xi in X])
        return (probs >= threshold).astype(int)


def QCNN() -> QCNNModel:
    """Factory returning a configured QCNNModel."""
    return QCNNModel()
