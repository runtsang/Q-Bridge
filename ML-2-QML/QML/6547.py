"""Quantum‑only implementation of QCNNEnhanced using Pennylane.

This module provides a minimal class that implements the quantum
circuit described in the seed.  It is intended for standalone use
without the classical feature extractor.
"""

import pennylane as qml
import torch
import numpy as np
from typing import Tuple

class QCNNEnhanced:
    """Quantum node for QCNN.

    The class implements the same ansatz as the hybrid model but
    without the preceding classical feature extractor.  It is
    callable like a function and returns the expectation value of
    PauliZ on the first qubit.
    """
    def __init__(self,
                 num_wires: int = 8,
                 num_layers: int = 3,
                 device: str = "default.qubit"):
        self.dev = qml.device(device, wires=num_wires)
        self.num_layers = num_layers
        self.num_wires = num_wires

        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            for i in range(num_wires):
                qml.RX(x[i], wires=i)
            for layer in range(num_layers):
                for i in range(num_wires):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(0, num_wires - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def __call__(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self.circuit(x, weights)

def QCNN() -> QCNNEnhanced:
    """Return a quantum‑only QCNNEnhanced instance."""
    return QCNNEnhanced()

__all__ = ["QCNN", "QCNNEnhanced"]
