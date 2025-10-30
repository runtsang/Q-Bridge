"""Quantum kernel with a trainable variational layer before measurement."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data and applies a trainable linear bias."""
    def __init__(self, func_list, n_wires: int = 4, trainable: bool = False) -> None:
        super().__init__()
        self.func_list = func_list
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.trainable = trainable
        if trainable:
            # Linear bias applied to the amplitude before measurement
            self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_buffer("bias", torch.zeros(1, dtype=torch.float32))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz with bias."""
    def __init__(self, trainable: bool = False) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ],
            n_wires=self.n_wires,
            trainable=trainable
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Apply bias before measurement
        amplitude = self.q_device.states.view(-1)[0]
        return torch.abs(amplitude + self.ansatz.bias)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], trainable: bool = False) -> np.ndarray:
    """Evaluate the Gram matrix between datasets a and b."""
    kernel = Kernel(trainable=trainable)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
