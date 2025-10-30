"""Quantum kernel construction with advanced parameter handling and hybrid support."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuantumKernelAnsatz(tq.QuantumModule):
    """Programmable quantum kernel ansatz with support for dynamic gate lists."""

    def __init__(self, func_list: List[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class HybridQuantumKernel(tq.QuantumModule):
    """Hybrid kernel that blends classical RBF and quantum circuit evaluations."""

    def __init__(self, *, gamma: float = 1.0, func_list: List[dict] | None = None, n_wires: int = 4):
        super().__init__()
        self.gamma = gamma
        if func_list is None:
            func_list = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        self.quantum_ansatz = QuantumKernelAnsatz(func_list)
        self.q_device = tq.QuantumDevice(n_wires=n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Classical RBF part
        diff = x - y
        class_val = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

        # Quantum part
        self.quantum_ansatz(self.q_device, x, y)
        quantum_val = torch.abs(self.q_device.states.view(-1)[0])

        # Weighted combination
        return 0.7 * class_val + 0.3 * quantum_val


def hybrid_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], *, gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using the hybrid quantum kernel."""
    kernel = HybridQuantumKernel(gamma=gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["QuantumKernelAnsatz", "HybridQuantumKernel", "hybrid_kernel_matrix"]
