"""Hybrid quantum‑classical QCNN using PennyLane."""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from pennylane import qnode
from pennylane.optimize import AdamOptimizer
from typing import Tuple


def angle_embedding(x: np.ndarray, wires: Tuple[int,...]) -> None:
    """Feature map using angle embedding on each qubit."""
    for idx, wire in enumerate(wires):
        qml.AngleEmbedding(x[idx], wires=[wire], rotation="Y")


def conv_layer(params: np.ndarray, wires: Tuple[int,...]) -> None:
    """Two‑qubit convolutional layer with a parameterized entangling block."""
    for i in range(0, len(wires), 2):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.RY(params[i], wires=wires[i])
        qml.RZ(params[i + 1], wires=wires[i + 1])
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def pooling_layer(params: np.ndarray, wires: Tuple[int,...]) -> None:
    """Pooling by measuring expectation of Z on each qubit."""
    for i, wire in enumerate(wires):
        qml.RZ(params[i], wires=wire)
        qml.PauliZ(wire)


def qc_cnn_circuit(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Full QCNN circuit combining feature map, convolutional, and pooling layers."""
    num_qubits = len(x)
    wires = tuple(range(num_qubits))

    # Feature map
    angle_embedding(x, wires)

    # First convolution & pooling
    conv_layer(weights[:num_qubits * 2], wires)
    pooling_layer(weights[num_qubits * 2: num_qubits * 3], wires)

    # Second convolution & pooling on reduced qubit set
    reduced_wires = wires[: num_qubits // 2]
    conv_layer(weights[num_qubits * 3: num_qubits * 5], reduced_wires)
    pooling_layer(weights[num_qubits * 5: num_qubits * 6], reduced_wires)

    # Final measurement
    return qml.expval(qml.PauliZ(0))


dev = qml.device("default.qubit", wires=8)


@qml.qnode(dev, interface="numpy")
def qcnn_qnode(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return qc_cnn_circuit(x, weights)


def QCNN(num_qubits: int = 8, seed: int = 42) -> Tuple[qml.QNode, np.ndarray]:
    """Return a QNode and initialized weight array for the QCNN."""
    np.random.seed(seed)
    # Each convolution uses 2 params per qubit, pooling uses 1 per qubit
    num_params = num_qubits * 6  # 3 layers of conv+pool
    weights = 0.01 * np.random.randn(num_params)
    return qcnn_qnode, weights


def train_qcnn(
    data: np.ndarray,
    labels: np.ndarray,
    epochs: int = 200,
    lr: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """Simple training loop for the QCNN using Adam."""
    qnode, weights = QCNN()
    opt = AdamOptimizer(lr)
    loss_fn = lambda y, t: np.mean((y - t) ** 2)

    for epoch in range(epochs):
        def loss_fn_wrapper(w):
            preds = np.array([qnode(x, w) for x in data])
            return loss_fn(preds, labels)

        weights = opt.step(loss_fn_wrapper, weights)
        if epoch % 20 == 0:
            preds = np.array([qnode(x, weights) for x in data])
            loss = loss_fn(preds, labels)
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

    return weights, loss
