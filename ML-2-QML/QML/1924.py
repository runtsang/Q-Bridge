"""Hybrid quantum‑classical QCNN implemented with Pennylane."""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QCNNEnhanced:
    """
    Quantum‑classical QCNN: a variational circuit on 8 qubits with
    a Z‑feature map and a 3‑layer ansatz. Supports batched
    evaluation and differentiable parameters via Pennylane's
    autograd interface.
    """
    def __init__(self, dev: qml.Device | None = None, seed: int = 123):
        self.dev = dev or qml.device("default.qubit", wires=8)
        self.seed = seed
        np.random.seed(self.seed)
        self.num_params = 3 * 8 * 3  # 3 rotations per qubit per layer, 3 layers
        self.weights = np.random.randn(self.num_params)
        self.qnode = self._build_qnode()
        self.batch_qnode = qml.batch(self.qnode)

    def _feature_map(self, x: np.ndarray) -> None:
        """Z‑feature map: RZ on each qubit followed by nearest‑neighbour CZ."""
        for i, val in enumerate(x):
            qml.RZ(val, wires=i)
        for i in range(len(x) - 1):
            qml.CZ(wires=[i, i + 1])

    def _ansatz(self, params: np.ndarray) -> None:
        """3‑layer rotation‑entanglement ansatz."""
        idx = 0
        for _ in range(3):
            for i in range(8):
                qml.RX(params[idx], wires=i); idx += 1
                qml.RY(params[idx], wires=i); idx += 1
                qml.RZ(params[idx], wires=i); idx += 1
            # Entangle neighbouring qubits
            for i in range(7):
                qml.CZ(wires=[i, i + 1])

    def _qnode(self, x):
        self._feature_map(x)
        self._ansatz(self.weights)
        return qml.expval(qml.PauliZ(0))

    def _build_qnode(self):
        return qml.QNode(self._qnode, self.dev, interface="autograd")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the QCNN on a batch of inputs.

        Args:
            x (np.ndarray): Shape (batch_size, 8) where each element is a real
                           number used as the rotation angle for the feature map.

        Returns:
            np.ndarray: Shape (batch_size,) with expectation values for qubit 0.
        """
        return self.batch_qnode(x)

    def set_weights(self, new_weights: np.ndarray) -> None:
        """Set new variational parameters."""
        if new_weights.shape!= self.weights.shape:
            raise ValueError("Weight shape mismatch.")
        self.weights = new_weights

    def get_weights(self) -> np.ndarray:
        """Return current variational parameters."""
        return self.weights


def QCNNEnhancedFactory() -> QCNNEnhanced:
    """
    Factory returning a ready‑to‑train :class:`QCNNEnhanced` instance.
    """
    return QCNNEnhanced()


__all__ = ["QCNNEnhanced", "QCNNEnhancedFactory"]
