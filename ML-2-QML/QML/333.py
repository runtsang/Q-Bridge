"""
QCNNHybrid: a quantum convolutional neural network built with PennyLane.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import qnode
from pennylane import Device
from typing import Tuple


class QCNNHybrid:
    """
    Quantum convolutional neural network.

    Parameters
    ----------
    dev_name : str, default "default.qubit"
        PennyLane device name.
    wires : int, default 8
        Number of qubits / input features.
    """

    def __init__(self, dev_name: str = "default.qubit", wires: int = 8) -> None:
        self.dev: Device = qml.device(dev_name, wires=wires)
        self.wires = wires

        # Parameter initialization
        self.feature_params = pnp.random.randn(wires) * 0.1
        self.conv_params = pnp.random.randn(wires * 3) * 0.1
        self.pool_params = pnp.random.randn((wires // 2) * 3) * 0.1
        self.ansatz_params = pnp.random.randn(wires * 3) * 0.1

        # QNode with autograd interface
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, inputs: np.ndarray,
                 feature_params: np.ndarray,
                 conv_params: np.ndarray,
                 pool_params: np.ndarray,
                 ansatz_params: np.ndarray) -> np.ndarray:
        """Parameterized quantum circuit implementing QCNN layers."""
        # ---------- Feature Map ----------
        for i in range(self.wires):
            qml.Hadamard(wires=i)
            qml.RZ(inputs[i], wires=i)

        # ---------- Convolutional Layer ----------
        def conv_layer(start: int, end: int, params: np.ndarray) -> None:
            """Apply a two‑qubit convolution block to qubits [start, end]."""
            # local 3‑parameter block
            qml.RZ(params[0], wires=start)
            qml.RY(params[1], wires=end)
            qml.CNOT(wires=[end, start])
            qml.RY(params[2], wires=end)

        # First convolution over pairs (0,1),(2,3),(4,5),(6,7)
        idx = 0
        for pair in [(0, 1), (2, 3), (4, 5), (6, 7)]:
            conv_layer(*pair, conv_params[idx:idx + 3])
            idx += 3

        # ---------- Pooling Layer ----------
        def pool_layer(source: int, sink: int, params: np.ndarray) -> None:
            """Apply a two‑qubit pooling block."""
            qml.RZ(params[0], wires=source)
            qml.RY(params[1], wires=sink)
            qml.CNOT(wires=[sink, source])
            qml.RY(params[2], wires=sink)

        # Pool (0,2) and (1,3) to reduce to 4 qubits
        idx = 0
        for src, snk in [(0, 2), (1, 3)]:
            pool_layer(src, snk, pool_params[idx:idx + 3])
            idx += 3

        # ---------- Second Convolution ----------
        idx = 0
        for pair in [(0, 1), (2, 3)]:
            conv_layer(*pair, conv_params[idx:idx + 3])
            idx += 3

        # ---------- Second Pooling ----------
        pool_layer(0, 2, pool_params[idx:idx + 3])

        # ---------- Third Convolution ----------
        conv_layer(0, 1, conv_params[idx:idx + 3])

        # ---------- Third Pooling ----------
        pool_layer(0, 1, pool_params[idx:idx + 3])

        # ---------- Ansatz ----------
        for i in range(self.wires):
            qml.RZ(ansatz_params[i], wires=i)

        # ---------- Measurement ----------
        return qml.expval(qml.PauliZ(0))

    def forward(self, inputs: np.ndarray) -> float:
        """Evaluate the QCNN on a single input vector."""
        return float(self.qnode(inputs,
                                self.feature_params,
                                self.conv_params,
                                self.pool_params,
                                self.ansatz_params))

    def parameters(self) -> Tuple[np.ndarray,...]:
        """Return all trainable parameters."""
        return (self.feature_params,
                self.conv_params,
                self.pool_params,
                self.ansatz_params)

    def set_parameters(self, params: Tuple[np.ndarray,...]) -> None:
        """Set all trainable parameters."""
        (self.feature_params,
         self.conv_params,
         self.pool_params,
         self.ansatz_params) = params

def QCNNHybrid() -> QCNNHybrid:
    """Factory returning a ready‑to‑train instance."""
    return QCNNHybrid()

__all__ = ["QCNNHybrid", "QCNNHybrid"]
