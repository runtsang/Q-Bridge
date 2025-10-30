"""Quantum kernel construction using TorchQuantum ansatz with trainable parameters."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torch import nn


class TrainableKernalAnsatz(tq.QuantumModule):
    """Encodes classical data with a trainable rotation amplitude and learns a phase shift."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Learnable amplitude for each wire
        self.amplitude = nn.Parameter(torch.ones(n_wires, dtype=torch.float32))
        # List of parameter‑dependent rotation functions
        self.func_list = [
            {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Apply forward data encoding
        for info in self.func_list:
            params = x[:, info["input_idx"]] * self.amplitude[info["input_idx"][0]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Apply backward data encoding with negated parameters
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] * self.amplitude[info["input_idx"][0]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class TrainableKernel(tq.QuantumModule):
    """Quantum kernel that learns rotation amplitudes and a global phase."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = TrainableKernalAnsatz(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return the absolute value of the first basis state amplitude
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = TrainableKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["TrainableKernalAnsatz", "TrainableKernel", "kernel_matrix"]
