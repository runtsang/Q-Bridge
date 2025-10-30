"""Quantum convolutional neural network implemented with PennyLane.

The circuit follows the same logical structure as the original Qiskit
implementation but adds a classical post‑processing layer and uses
parameter‑shifting for efficient gradient evaluation.  The model
is fully differentiable and can be trained with any PyTorch optimiser.
"""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
import torch
from torch import nn

# Device with 8 qubits
dev = qml.device("default.qubit", wires=8)


def z_feature_map(x: np.ndarray) -> None:
    """Z‑feature map from Qiskit mapped to PennyLane."""
    for i, val in enumerate(x):
        qml.RZ(val, wires=i)


def conv_block(params: np.ndarray, wires: list[int]) -> None:
    """Two‑qubit convolution gate with 3 parameters per pair."""
    n_pairs = len(wires) // 2
    for i in range(n_pairs):
        idx = 3 * i
        w1, w2 = wires[2 * i], wires[2 * i + 1]
        qml.RZ(params[idx], wires=w1)
        qml.RY(params[idx + 1], wires=w2)
        qml.CNOT(w1, w2)
        qml.RY(params[idx + 2], wires=w2)


def pool_block(params: np.ndarray, wires: list[int]) -> None:
    """Two‑qubit pooling gate with 3 parameters per pair."""
    n_pairs = len(wires) // 2
    for i in range(n_pairs):
        idx = 3 * i
        w1, w2 = wires[2 * i], wires[2 * i + 1]
        qml.RZ(params[idx], wires=w1)
        qml.RY(params[idx + 1], wires=w2)
        qml.CNOT(w1, w2)
        qml.RY(params[idx + 2], wires=w2)


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def qcnn_circuit(inputs: torch.Tensor,
                 conv1_w: torch.Tensor,
                 pool1_w: torch.Tensor,
                 conv2_w: torch.Tensor,
                 pool2_w: torch.Tensor,
                 conv3_w: torch.Tensor,
                 pool3_w: torch.Tensor) -> torch.Tensor:
    """Full QCNN circuit with 3 conv/pool layers."""
    z_feature_map(inputs)

    conv_block(conv1_w, list(range(8)))
    pool_block(pool1_w, list(range(0, 8, 2)))

    conv_block(conv2_w, list(range(4, 8)))
    pool_block(pool2_w, list(range(0, 4, 2)))

    conv_block(conv3_w, list(range(6, 8)))
    pool_block(pool3_w, list(range(6, 8, 2)))

    return qml.expval(qml.PauliZ(0))


class QCNNHybrid(nn.Module):
    """Hybrid QCNN with a classical post‑processing linear layer.

    The quantum part outputs a single expectation value which is then
    fed through a learnable linear layer to produce the final scalar
    prediction.  This mirrors the classical QCNN head.
    """

    def __init__(self) -> None:
        super().__init__()
        # Parameter tensors for each layer
        self.conv1_w = nn.Parameter(torch.randn(12))
        self.pool1_w = nn.Parameter(torch.randn(6))
        self.conv2_w = nn.Parameter(torch.randn(6))
        self.pool2_w = nn.Parameter(torch.randn(3))
        self.conv3_w = nn.Parameter(torch.randn(3))
        self.pool3_w = nn.Parameter(torch.randn(1))

        # Classical post‑processing
        self.classifier = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        q_out = qcnn_circuit(x,
                             self.conv1_w,
                             self.pool1_w,
                             self.conv2_w,
                             self.pool2_w,
                             self.conv3_w,
                             self.pool3_w)
        # Ensure compatible shape for the linear layer
        q_out = q_out.unsqueeze(-1)
        return torch.sigmoid(self.classifier(q_out))

def QCNN() -> QCNNHybrid:
    """Factory returning the configured :class:`QCNNHybrid`."""
    return QCNNHybrid()


__all__ = ["QCNN", "QCNNHybrid"]
