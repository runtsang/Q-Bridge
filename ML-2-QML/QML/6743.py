"""Quantum kernel module using TorchQuantum."""
from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np
from typing import Sequence

class QuantumKernel(tq.QuantumModule):
    """Parameterized ansatz for quantum kernel evaluation."""
    def __init__(self, func_list: list[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice,
                x: torch.Tensor,
                y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernelModule(tq.QuantumModule):
    """Wrapper that returns the absolute overlap as kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernel(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def quantum_kernel_matrix(a: Sequence[torch.Tensor],
                          b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute quantum kernel Gram matrix."""
    kernel = QuantumKernelModule()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = [
    "QuantumKernel",
    "QuantumKernelModule",
    "quantum_kernel_matrix",
]
