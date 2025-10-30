"""
Hybrid quantum‑classical QCNN with parameter‑shift training and Adam optimisation.

The implementation builds a layered circuit comprising a Z‑feature map, convolutional and pooling blocks,
and trains the ansatz weights using a hybrid loss function.  The class exposes a simple
``predict`` and ``fit`` API that mirrors the classical counterpart.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
from pennylane.measurements import expectation
from typing import Tuple, List

__all__ = ["QCNNHybrid", "QCNN"]


def _conv_circuit(qubits: List[int], params: pnp.ndarray) -> qml.QNode:
    """Two‑qubit convolution block used throughout the circuit."""
    def circuit():
        for q in qubits:
            qml.RZ(-np.pi / 2, wires=q)
        qml.CNOT(wires=[qubits[1], qubits[0]])
        qml.RZ(params[0], wires=qubits[0])
        qml.RY(params[1], wires=qubits[1])
        qml.CNOT(wires=[qubits[0], qubits[1]])
        qml.RY(params[2], wires=qubits[1])
        qml.CNOT(wires=[qubits[1], qubits[0]])
        qml.RZ(np.pi / 2, wires=qubits[0])
    return qml.QNode(circuit, device)


def _pool_circuit(qubits: List[int], params: pnp.ndarray) -> qml.QNode:
    """Two‑qubit pooling block used throughout the circuit."""
    def circuit():
        for q in qubits:
            qml.RZ(-np.pi / 2, wires=q)
        qml.CNOT(wires=[qubits[1], qubits[0]])
        qml.RZ(params[0], wires=qubits[0])
        qml.RY(params[1], wires=qubits[1])
        qml.CNOT(wires=[qubits[0], qubits[1]])
        qml.RY(params[2], wires=qubits[1])
    return qml.QNode(circuit, device)


class QCNNHybrid:
    """
    Quantum Convolutional Neural Network with hybrid training.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits (default 8).
    shots : int
        Number of shots for expectation estimation (default 1024).
    device_name : str
        Pennylane device name (default 'default.qubit').

    Notes
    -----
    The circuit consists of:
    * A Z‑feature map encoding the classical input.
    * Three convolutional layers interleaved with pooling layers.
    * All weights are trainable parameters of the ansatz.
    """

    def __init__(self, num_qubits: int = 8, shots: int = 1024, device_name: str = "default.qubit") -> None:
        self.num_qubits = num_qubits
        self.shots = shots
        self.device = qml.device(device_name, wires=num_qubits, shots=shots)

        # Feature map
        self.feature_map = qml.feature_map.ZFeatureMap(num_qubits, reps=1)
        # Ansatz parameters
        self.ansatz_params = pnp.random.uniform(0, 2 * np.pi, self._num_ansatz_params())
        # Observable
        self.observable = qml.PauliZ(0)

    def _num_ansatz_params(self) -> int:
        """Count the total number of trainable parameters."""
        # 3 params per conv/pool block per pair of qubits
        conv_blocks = 3  # first layer (4 qubits), second (2), third (1)
        pool_blocks = 2  # after conv1 and conv2
        total_pairs = (4 + 2 + 1) + (2 + 1)  # conv pairs + pool pairs
        return total_pairs * 3

    def _build_circuit(self, x: np.ndarray) -> qml.QNode:
        """Construct the full parameterised circuit for a single data point."""
        @qml.qnode(self.device, interface="autograd")
        def circuit():
            # Step 1: feature map
            self.feature_map(x)
            # Step 2: ansatz (convolution + pooling)
            # First convolutional layer
            for i in range(0, 8, 2):
                idx = i // 2
                params = self.ansatz_params[idx * 3 : (idx + 1) * 3]
                _conv_circuit([i, i + 1], params)()
            # First pooling layer
            for i in [0, 1]:
                idx = 4 + i
                params = self.ansatz_params[idx * 3 : (idx + 1) * 3]
                _pool_circuit([i, i + 1], params)()
            # Second convolutional layer
            for i in range(4, 8, 2):
                idx = 6 + (i - 4) // 2
                params = self.ansatz_params[idx * 3 : (idx + 1) * 3]
                _conv_circuit([i, i + 1], params)()
            # Second pooling layer
            for i in [0, 1]:
                idx = 8 + i
                params = self.ansatz_params[idx * 3 : (idx + 1) * 3]
                _pool_circuit([i, i + 1], params)()
            # Third convolutional layer
            idx = 10
            params = self.ansatz_params[idx * 3 : (idx + 1) * 3]
            _conv_circuit([6, 7], params)()
            # Third pooling layer
            idx = 11
            params = self.ansatz_params[idx * 3 : (idx + 1) * 3]
            _pool_circuit([6, 7], params)()
            return expectation(self.observable)
        return circuit

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the raw expectation values for a batch of inputs."""
        preds = []
        for x in X:
            circuit = self._build_circuit(x)
            preds.append(circuit())
        return np.array(preds)

    def loss(self, preds: np.ndarray, y: np.ndarray) -> float:
        """Binary cross‑entropy loss."""
        eps = 1e-12
        preds = np.clip(preds, eps, 1 - eps)
        return -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 32,
    ) -> List[float]:
        """Hybrid training loop using Adam optimisation."""
        optimizer = AdamOptimizer(lr)
        history = []

        for epoch in range(1, epochs + 1):
            # Shuffle the data
            perm = np.random.permutation(len(X))
            X, y = X[perm], y[perm]
            epoch_loss = 0.0
            for i in range(0, len(X), batch_size):
                xb, yb = X[i : i + batch_size], y[i : i + batch_size]
                # Compute gradients
                grads = optimizer.gradient(lambda w: self.loss(self.predict(xb), yb), self.ansatz_params)
                # Update parameters
                self.ansatz_params = optimizer.apply_gradient(self.ansatz_params, grads)
                epoch_loss += self.loss(self.predict(xb), yb)
            epoch_loss /= (len(X) // batch_size)
            history.append(epoch_loss)
            print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f}")

        return history


def QCNN() -> QCNNHybrid:
    """Factory returning a ready‑to‑train :class:`QCNNHybrid` instance."""
    return QCNNHybrid()
