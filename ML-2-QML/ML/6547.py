"""Hybrid classical‑quantum convolutional network with joint training.

The original seed defined a purely classical stack of linear layers that
mimicked the QCNN architecture.  Here we expose a *feature‑extractor* that can be
trained separately or jointly with the quantum block.  The quantum block is
implemented as a parameter‑ized Pennylane QNode that returns a real‑valued
measurement.  The two blocks are wrapped in a single ``torch.nn.Module`` so
that gradients flow from the quantum circuit to the classical parameters
via the ``torch.autograd`` interface provided by pennylane.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from typing import Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Classical feature extractor
# ──────────────────────────────────────────────────────────────────────────────
class QCNNFeatureExtractor(nn.Module):
    """Linear layers that emulate the original QCNN feature‑map.

    The architecture matches the seed but is exposed as a *module* so that
    it can be trained independently or as part of the hybrid model.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# ──────────────────────────────────────────────────────────────────────────────
# Hybrid model
# ──────────────────────────────────────────────────────────────────────────────
class QCNNEnhanced(nn.Module):
    """Hybrid classical‑quantum convolutional network.

    The model consists of a classical feature extractor followed by a
    Pennylane QNode that implements a parameterised ansatz.  The forward
    pass returns the expectation value of PauliZ on the first qubit.
    """
    def __init__(self,
                 input_dim: int = 8,
                 hidden_dim: int = 16,
                 num_layers: int = 3,
                 device: str = "default.qubit",
                 wires: int = 8) -> None:
        super().__init__()
        self.feature_extractor = QCNNFeatureExtractor(input_dim, hidden_dim)
        self.num_layers = num_layers
        self.wires = wires
        self.dev = qml.device(device, wires=wires)

        # Initialise random weights for the quantum ansatz
        self.weight_shape = (num_layers, wires)
        self.weights = nn.Parameter(torch.randn(self.weight_shape))

        # Create the QNode
        self.qnode = self._make_qnode()

    def _make_qnode(self) -> qml.QNode:
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Feature map: RX rotations from the classical feature map
            for i in range(self.wires):
                qml.RX(x[i], wires=i)

            # Ansatz: alternating RY rotations and CNOTs
            for layer in range(self.num_layers):
                for i in range(self.wires):
                    qml.RY(weights[layer, i], wires=i)
                # Pairwise CNOTs
                for i in range(0, self.wires - 1, 2):
                    qml.CNOT(wires=[i, i + 1])

            return qml.expval(qml.PauliZ(0))

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through classical part
        h = self.feature_extractor(x)
        # Forward through quantum part
        return self.qnode(h, self.weights)

def QCNN() -> QCNNEnhanced:
    """Factory returning the configured :class:`QCNNEnhanced`."""
    return QCNNEnhanced()

__all__ = ["QCNN", "QCNNEnhanced"]
