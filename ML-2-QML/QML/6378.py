"""Quantum kernel implementation using TorchQuantum.

The :class:`QuantumKernelMethod__gen192` module:
* Encodes two classical vectors with a small variational circuit.
* Adds a single trainable rotation to the ansatz.
* Provides a reproducible seed and a helper to compute a Gram matrix.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torch import nn

__all__ = ["QuantumKernelMethod__gen192", "QuantumAnsatz"]

class QuantumAnsatz(tq.QuantumModule):
    """Variational ansatz with a single trainable rotation."""

    def __init__(self, n_wires: int = 4, seed: int = 42) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.layers = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.q_device.reset_states(x.shape[0])
        # Encode data x
        for layer in self.layers:
            params = x[:, layer["input_idx"]] if layer["func"] in ["ry", "rz"] else None
            func_name_dict[layer["func"]](self.q_device, wires=layer["wires"], params=params)
        # Apply trainable rotation
        func_name_dict["ry"](self.q_device, wires=[0], params=self.theta)
        # Encode data y with negative sign
        for layer in reversed(self.layers):
            params = -y[:, layer["input_idx"]] if layer["func"] in ["ry", "rz"] else None
            func_name_dict[layer["func"]](self.q_device, wires=layer["wires"], params=params)
        return torch.abs(self.q_device.states.view(-1)[0])

class QuantumKernelMethod__gen192(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""

    def __init__(self, n_wires: int = 4, seed: int = 42) -> None:
        super().__init__()
        self.ansatz = QuantumAnsatz(n_wires, seed)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

    def kernel_matrix(self, a: list[torch.Tensor], b: list[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two datasets."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])
