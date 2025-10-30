"""Quantum kernel construction using a parameterised TorchQuantum ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum import op_name_dict

class KernalAnsatz(tq.QuantumModule):
    """Parameterized quantum ansatz for kernel evaluation."""
    def __init__(self, n_wires: int, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.gate_list = []
        for d in range(depth):
            for w in range(n_wires):
                self.gate_list.append({"func": "ry", "wires": [w], "input_idx": [w]})
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for gate in self.gate_list:
            params = x[:, gate["input_idx"]] if op_name_dict[gate["func"]].num_params else None
            func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=params)
        for gate in reversed(self.gate_list):
            params = -y[:, gate["input_idx"]] if op_name_dict[gate["func"]].num_params else None
            func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel module."""
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(self.n_wires, depth)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix using the quantum kernel."""
        return np.array([[self(x, y).item() for y in b] for x in a])

class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel with trainable parameters and Gram matrix evaluation."""
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.kernel = Kernel(n_wires, depth)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return self.kernel.kernel_matrix(a, b)

__all__ = ["KernalAnsatz", "Kernel", "QuantumKernelMethod"]
