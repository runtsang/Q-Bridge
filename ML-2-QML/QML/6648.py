"""Quantum kernel construction using TorchQuantum ansatz with tunable depth."""

from __future__ import annotations

from typing import Sequence, List

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumAnsatz(tq.QuantumModule):
    """Variational ansatz with parameter‑sharing and depth control."""

    def __init__(self, n_wires: int, depth: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        # Create a list of gates for each layer
        self.layers = []
        for _ in range(depth):
            layer = [
                {"func": "ry", "wires": [i], "input_idx": [i]}
                for i in range(n_wires)
            ]
            self.layers.append(layer)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Forward encoding
        for layer in self.layers:
            for gate in layer:
                params = x[:, gate["input_idx"]] if tq.op_name_dict[gate["func"]].num_params else None
                func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=params)
        # Reverse encoding with negative parameters
        for layer in reversed(self.layers):
            for gate in layer:
                params = -y[:, gate["input_idx"]] if tq.op_name_dict[gate["func"]].num_params else None
                func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=params)

class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel module with tunable depth and shared parameters."""

    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = QuantumAnsatz(n_wires, depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return the overlap between the first state of the device
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], depth: int = 2) -> np.ndarray:
    """Evaluate the Gram matrix using the quantum kernel with specified depth."""
    kernel = QuantumKernelMethod(depth=depth)
    return np.array([[kernel(x, y).item() for y in b] for x in a])
