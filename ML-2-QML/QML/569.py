"""Quantum kernel construction with a trainable variational ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum.layers import QuantumModule

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates and
    learns optimal feature‑map weights via a small neural network."""
    def __init__(self, n_wires: int, n_layers: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.params = nn.Parameter(torch.randn(n_layers, n_wires))

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x
        for layer in range(self.n_layers):
            for wire in range(self.n_wires):
                idx = layer * self.n_wires + wire
                param = self.params[layer, wire] * x[:, wire]
                tq.ry(q_device, wire=wire, param=param)
        # Encode y (reverse sign)
        for layer in reversed(range(self.n_layers)):
            for wire in range(self.n_wires):
                idx = layer * self.n_wires + wire
                param = -self.params[layer, wire] * y[:, wire]
                tq.ry(q_device, wire=wire, param=param)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a trainable variational ansatz."""
    def __init__(self, n_wires: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(self.n_wires, self.n_layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])
