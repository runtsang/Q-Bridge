"""Hybrid quantum kernel that encodes data and sampler-derived weights."""

from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np
from typing import Sequence

class KernalAnsatz(tq.QuantumModule):
    def __init__(self, n_wires: int = 4, weight_dim: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.weight_dim = weight_dim
        self.weight = torch.zeros(weight_dim)
        # Build a list of gates: data encoding, weight encoding, inverse data
        self.func_list = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
            {"weight_idx": [0], "func": "ry", "wires": [0]},
            {"weight_idx": [1], "func": "ry", "wires": [1]},
            {"weight_idx": [2], "func": "ry", "wires": [2]},
            {"weight_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        batch = x.shape[0]
        q_device.reset_states(batch)
        # Encode data
        for info in self.func_list[:4]:
            params = x[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Encode weights
        for info in self.func_list[4:]:
            params = self.weight[info["weight_idx"]].expand(batch)
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Encode inverse data
        for info in reversed(self.func_list[:4]):
            params = -y[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    def __init__(self, n_wires: int = 4, weight_dim: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(n_wires, weight_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], weight: torch.Tensor) -> np.ndarray:
    """
    Evaluate the Gram matrix using the hybrid quantum kernel.
    The weight vector should be produced by the classical sampler.
    """
    kernel = Kernel()
    kernel.ansatz.weight = weight
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
