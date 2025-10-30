"""Hybrid RBF‑quantum kernel module – quantum side."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class VariationalAnsatz(tq.QuantumModule):
    """Variational circuit with trainable parameters and a simple data encoding."""
    def __init__(self, n_wires: int, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.params = nn.Parameter(torch.randn(depth, n_wires, 3))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for d in range(self.depth):
            for w in range(self.n_wires):
                # data encoding (Ry)
                func_name_dict["ry"](q_device, wires=[w], params=x[:, w:w+1])
                # variational rotation (Rz)
                func_name_dict["rz"](q_device, wires=[w], params=self.params[d, w, 0:1])
            # entanglement layer (CNOT chain)
            for w in range(self.n_wires - 1):
                func_name_dict["cx"](q_device, wires=[w, w + 1])
        # reverse encoding for y
        for d in reversed(range(self.depth)):
            for w in range(self.n_wires):
                func_name_dict["rz"](q_device, wires=[w], params=-self.params[d, w, 0:1])
                func_name_dict["ry"](q_device, wires=[w], params=y[:, w:w+1])


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a trainable variational ansatz."""
    def __init__(self, n_wires: int = 4, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = VariationalAnsatz(self.n_wires, depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int = 4, depth: int = 2) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b`` using a variational circuit."""
    kernel = Kernel(n_wires, depth)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["VariationalAnsatz", "Kernel", "kernel_matrix"]
