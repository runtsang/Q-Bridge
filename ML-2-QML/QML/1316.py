"""Quantum‑convolutional neural network implemented with Pennylane.

The :class:`QCNNModel` class wraps a variational circuit that
implements the same logical depth as the classical seed but
leverages automatic differentiation in Pennylane.  It exposes a
`train_step` method that accepts PyTorch tensors and integrates
with the same optimiser interface used by the classical model.
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from typing import Iterable, List

# 8‑qubit device (can be swapped for a real backend)
dev = qml.device("default.qubit", wires=8)

def _feature_map(x: torch.Tensor) -> None:
    """Simple RY feature map."""
    for i, wire in enumerate(range(dev.num_wires)):
        qml.RY(x[i], wires=wire)

def _ansatz_layer(params: torch.Tensor, layer_idx: int) -> None:
    """Entangling ansatz layer with 3‑parameter rotation per qubit."""
    for i, wire in enumerate(range(dev.num_wires)):
        qml.RZ(params[layer_idx, i, 0], wires=wire)
        qml.RY(params[layer_idx, i, 1], wires=wire)
        qml.RZ(params[layer_idx, i, 2], wires=wire)
    # pairwise CNOT entanglement
    for i in range(0, dev.num_wires - 1, 2):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev, interface="torch")
def _qnode(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    _feature_map(x)
    for layer in range(params.shape[0]):
        _ansatz_layer(params, layer)
    # single‑qubit measurement
    return qml.expval(qml.PauliZ(0))

class QCNNModel(nn.Module):
    """
    Variational QCNN that mirrors the depth of the classical seed
    but uses Pennylane for parameter optimisation.  The model
    accepts an 8‑dimensional input vector and outputs a probability
    via a sigmoid activation.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # trainable parameters for the ansatz
        self.params = nn.Parameter(
            init_std * torch.randn(n_layers, n_qubits, 3)
        )
        # optional weight for the final sigmoid
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the variational circuit followed by a
        sigmoid activation to produce a probability.
        """
        y = _qnode(x, self.params)
        return torch.sigmoid(self.weight * y)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """
        One training step with automatic differentiation.
        """
        self.train()
        optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

def QCNN() -> QCNNModel:
    """
    Factory that returns a ready‑to‑train :class:`QCNNModel` instance.
    """
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
