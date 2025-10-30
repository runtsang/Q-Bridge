"""Quantum kernel construction using a variational TorchQuantum ansatz.

The new implementation introduces a trainable offset for each qubit
and a fixed ring‑shaped entanglement pattern.  This yields a richer
feature map while keeping the interface identical to the original
``Kernel`` class.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torch import nn


class KernalAnsatz(tq.QuantumModule):
    """Variational ansatz that encodes two classical vectors through
    a trainable RY layer followed by a fixed entanglement pattern."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Trainable offsets for each qubit
        self.offset = nn.Parameter(torch.randn(n_wires))
        # Entanglement pattern: adjacent CNOTs in a ring
        self.entanglement = [(i, (i + 1) % n_wires) for i in range(n_wires)]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice,
                x: torch.Tensor,
                y: torch.Tensor) -> None:
        # Encode the first vector
        for i in range(self.n_wires):
            tq.ry(q_device, wires=[i], params=x[:, i] + self.offset[i])
        # Entangle qubits
        for i, j in self.entanglement:
            tq.cnot(q_device, wires=[i, j])
        # Encode the negative of the second vector
        for i in range(self.n_wires):
            tq.ry(q_device, wires=[i], params=-y[:, i] + self.offset[i])
        # Reverse entanglement
        for i, j in reversed(self.entanglement):
            tq.cnot(q_device, wires=[i, j])


class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two variational states."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two collections of samples."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
