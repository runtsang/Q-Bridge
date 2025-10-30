# Quantum kernel construction using TorchQuantum ansatz with data re-uploading and augmentation.

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = ["QuantumKernelMethod", "kernel_matrix"]

class QuantumAnsatz(tq.QuantumModule):
    'Variational ansatz with data re-uploading and a small data-augmentation circuit.'
    def __init__(self, n_wires: int, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.params = torch.nn.Parameter(torch.randn(depth, n_wires))
        self.augment = [tq.ops.RY(0.1) for _ in range(n_wires)]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for d in range(self.depth):
            for i in range(self.n_wires):
                q_device.apply(tq.ops.RY, wires=[i], params=x[:, i] * self.params[d, i])
            for i in range(self.n_wires):
                q_device.apply(tq.ops.RZ, wires=[i], params=self.params[d, i])
            for aug in self.augment:
                q_device.apply(aug, wires=[i])
        for d in reversed(range(self.depth)):
            for i in range(self.n_wires):
                q_device.apply(tq.ops.RY, wires=[i], params=-y[:, i] * self.params[d, i])
            for i in range(self.n_wires):
                q_device.apply(tq.ops.RZ, wires=[i], params=self.params[d, i])
            for aug in self.augment:
                q_device.apply(aug, wires=[i])

class QuantumKernelMethod(tq.QuantumModule):
    'Quantum kernel evaluated via a variational ansatz.'
    def __init__(self, n_wires: int = 4, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumAnsatz(n_wires=self.n_wires, depth=self.depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int = 4, depth: int = 2) -> np.ndarray:
    'Evaluate the Gram matrix between datasets a and b using the variational ansatz.'
    kernel = QuantumKernelMethod(n_wires=n_wires, depth=depth)
    return np.array([[kernel(x, y).item() for y in b] for x in a])
