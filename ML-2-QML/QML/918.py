"""Quantum convolutional neural network implemented with PennyLane.

The implementation builds on the classical QCNN layout but replaces the
fixed parameterised layers with a trainable variational ansatz.
"""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from pennylane import qnn
import torch
from torch import nn

# Device: 8 qubits, using default simulator
dev = qml.device("default.qubit", wires=8)

def conv_circuit(wires, params):
    """Two‑qubit convolution unit."""
    qml.RZ(-np.pi/2, wires=wires[1])
    qml.CNOT(wires=wires[1:2])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=wires[0:1])
    qml.RY(params[2], wires=wires[1])
    qml.CNOT(wires=wires[1:2])
    qml.RZ(np.pi/2, wires=wires[0])

def pool_circuit(wires, params):
    """Two‑qubit pooling unit."""
    qml.RZ(-np.pi/2, wires=wires[1])
    qml.CNOT(wires=wires[1:2])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=wires[0:1])
    qml.RY(params[2], wires=wires[1])

@qml.qnode(dev, interface="torch")
def qcnn_qnode(inputs, weights):
    """Hybrid QNode: feature map + variational ansatz."""
    # Feature map: Z feature map
    for i, w in enumerate(inputs):
        qml.RZ(w, wires=i)

    idx = 0
    # First convolutional layer (4 pairs)
    for pair in [(0,1),(2,3),(4,5),(6,7)]:
        conv_circuit(pair, weights[idx:idx+3])
        idx += 3
    # First pooling layer
    for pair in [(0,1),(2,3),(4,5),(6,7)]:
        pool_circuit(pair, weights[idx:idx+3])
        idx += 3
    # Second convolutional layer (2 pairs)
    for pair in [(4,5),(6,7)]:
        conv_circuit(pair, weights[idx:idx+3])
        idx += 3
    # Second pooling layer
    for pair in [(4,5),(6,7)]:
        pool_circuit(pair, weights[idx:idx+3])
        idx += 3
    # Third convolutional layer (1 pair)
    conv_circuit([6,7], weights[idx:idx+3])
    idx += 3
    # Third pooling layer
    pool_circuit([6,7], weights[idx:idx+3])
    idx += 3

    # Measurement
    return qml.expval(qml.PauliZ(0))

class QCNNQML(nn.Module):
    """Hybrid QCNN model combining a PennyLane QNode with a classical readout."""

    def __init__(self, weight_shape: tuple[int,...] | None = None):
        super().__init__()
        # Total number of variational parameters: 42
        if weight_shape is None:
            weight_shape = (42,)
        self.weight_shape = weight_shape
        self.qnode = qcnn_qnode
        self.classifier = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input to 8 features
        x = x.view(-1, 8)
        # Quantum expectation values
        qout = self.qnode(x, torch.flatten(self.weight_shape))
        # Classical readout
        return torch.sigmoid(self.classifier(qout))

def QCNN() -> QCNNQML:
    """Return a default‑configured hybrid QCNN model."""
    return QCNNQML()

__all__ = ["QCNN", "QCNNQML"]
