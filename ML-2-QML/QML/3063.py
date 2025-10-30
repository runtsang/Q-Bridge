"""Hybrid quantum kernel using a variational QCNN ansatz."""
from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum import functional as func
from typing import Sequence
import numpy as np

class QCNNAnsatz(tq.QuantumModule):
    """Variational QCNN ansatz built with TorchQuantum gates."""
    def __init__(self, n_wires: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Variational weight parameters for the convolutional layers
        self.weight = tq.Parameter(torch.randn(n_wires * 3))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # Encode input x
        for i in range(self.n_wires):
            func.ry(q_device, wires=[i], params=x[0, i])

        # Apply convolutional layers with variational weights
        self._apply_conv_layer(q_device, self.weight)

        # Encode input y with negative sign to form the kernel overlap
        for i in range(self.n_wires):
            func.ry(q_device, wires=[i], params=-y[0, i])

        # Apply the same convolutional layers again
        self._apply_conv_layer(q_device, self.weight)

    @tq.static_support
    def _apply_conv_layer(self, q_device: tq.QuantumDevice, params: torch.Tensor) -> None:
        # params shape: (n_wires * 3,)
        for i in range(0, self.n_wires, 2):
            idx = i // 2
            p0 = params[idx * 3]
            p1 = params[idx * 3 + 1]
            p2 = params[idx * 3 + 2]
            # Convolutional sub‑circuit
            func.ry(q_device, wires=[i], params=p0)
            func.ry(q_device, wires=[i + 1], params=p1)
            func.cx(q_device, control=i, target=i + 1)
            func.ry(q_device, wires=[i + 1], params=p2)
            func.cx(q_device, control=i + 1, target=i)
            # Pooling sub‑circuit (simplified)
            func.ry(q_device, wires=[i], params=p0)
            func.ry(q_device, wires=[i + 1], params=p1)
            func.cx(q_device, control=i, target=i + 1)

class HybridKernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states
    through a variational QCNN ansatz."""
    def __init__(self, n_wires: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QCNNAnsatz(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: tensors of shape (batch, n_wires)
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # The kernel value is the absolute overlap with the initial state
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self(x, y).item() for y in b] for x in a])

__all__ = ["QCNNAnsatz", "HybridKernel"]
