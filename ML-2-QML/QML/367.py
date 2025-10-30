"""
QCNN – a Pennylane variational quantum circuit that mirrors the architecture
of the quantum convolutional neural network.  It consists of a
feature‑map layer, a stack of convolutional layers (entangling two‑qubit
rotations), and a pooling layer (controlled rotations that reduce the
effective number of wires).  The circuit returns the expectation value
of Pauli‑Z on wire 0, which can be used as a classification score.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Callable

__all__ = ["QCNN"]


class QCNN:
    """
    Quantum QCNN built with Pennylane.

    Parameters
    ----------
    n_qubits : int, default=8
        Number of qubits in the circuit.
    n_layers : int, default=3
        Number of convolution‑pooling layers.
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 3) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Weight tensors: shape (n_layers, n_qubits, 3) for RX,RZ,RY rotations
        self.weights = np.random.randn(n_layers, n_qubits, 3) * np.pi
        self.qnode = qml.QNode(self._circuit, self.dev)

    # ------------------------------------------------------------------
    # Feature map – simple RX‑RZ‑CNOT pattern
    # ------------------------------------------------------------------
    def _feature_map(self, x: np.ndarray) -> None:
        for i, val in enumerate(x):
            qml.RX(val, wires=i)
            qml.RZ(val, wires=i)
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

    # ------------------------------------------------------------------
    # Convolutional layer – two‑qubit rotations + CNOT
    # ------------------------------------------------------------------
    def _conv_layer(self, layer_idx: int) -> None:
        for q in range(0, self.n_qubits, 2):
            w = self.weights[layer_idx, q]
            qml.RX(w[0], wires=q)
            qml.RZ(w[1], wires=q)
            qml.RY(w[2], wires=q)

            w = self.weights[layer_idx, q + 1]
            qml.RX(w[0], wires=q + 1)
            qml.RZ(w[1], wires=q + 1)
            qml.RY(w[2], wires=q + 1)

            qml.CNOT(wires=[q, q + 1])

    # ------------------------------------------------------------------
    # Pooling layer – simple CNOT that discards one qubit per pair
    # ------------------------------------------------------------------
    def _pool_layer(self) -> None:
        for q in range(0, self.n_qubits, 2):
            qml.CNOT(wires=[q, q + 1])

    # ------------------------------------------------------------------
    # Variational circuit
    # ------------------------------------------------------------------
    def _circuit(self, x: np.ndarray, weights: np.ndarray) -> float:
        self._feature_map(x)
        for layer_idx in range(self.n_layers):
            self._conv_layer(layer_idx)
            self._pool_layer()
        # Classification output: expectation of Pauli‑Z on wire 0
        return qml.expval(qml.PauliZ(0))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> float:
        """
        Evaluate the QCNN on a single input vector.

        Parameters
        ----------
        x : np.ndarray
            Input of shape ``(n_qubits,)``.

        Returns
        -------
        float
            The raw prediction (expectation value of Pauli‑Z on wire 0).
        """
        return self.qnode(x, self.weights)

    def get_qnode(self) -> Callable[[np.ndarray, np.ndarray], float]:
        """
        Return the underlying Pennylane QNode for external optimisation.

        Returns
        -------
        Callable
            A function that takes ``(x, weights)`` and returns the prediction.
        """
        return self.qnode
