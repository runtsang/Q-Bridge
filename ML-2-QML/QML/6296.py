"""Quantum kernel construction using TorchQuantum ansatz with entanglement layers."""

from __future__ import annotations

from typing import Sequence, Iterable, Union
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
]

class KernalAnsatz(tq.QuantumModule):
    """Encodes data using a multi‑layer variational circuit."""
    def __init__(self, n_wires: int, n_layers: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Build a list of operations for each layer
        self.func_list: list[dict] = []
        for layer in range(n_layers):
            # Rotation layer
            for w in range(n_wires):
                self.func_list.append({"input_idx": [w], "func": "ry", "wires": [w]})
            # Entanglement layer (CNOT chain)
            for w in range(n_wires - 1):
                self.func_list.append({"input_idx": [], "func": "cnot", "wires": [w, w + 1]})

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x forward
        for info in self.func_list:
            if info["func"] == "cnot":
                func_name_dict[info["func"]](q_device, wires=info["wires"])
            else:
                params = x[:, info["input_idx"]]
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Encode y backward (inverse)
        for info in reversed(self.func_list):
            if info["func"] == "cnot":
                func_name_dict[info["func"]](q_device, wires=info["wires"])
            else:
                params = -y[:, info["input_idx"]]
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""
    def __init__(self, n_wires: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(n_wires=self.n_wires, n_layers=n_layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int = 4, n_layers: int = 2) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b`` using a quantum kernel."""
    kernel = Kernel(n_wires=n_wires, n_layers=n_layers)
    return np.array([[kernel(x, y).item() for y in b] for x in a])
