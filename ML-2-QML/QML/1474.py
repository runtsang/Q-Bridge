"""Quantum kernel construction using a variational TorchQuantum ansatz."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
import torchquantum as tq
from torch import nn

class KernalAnsatz(tq.QuantumModule):
    """Variational quantum kernel with trainable parameters."""
    def __init__(self, n_wires: int = 4, n_layers: int = 1):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Trainable rotation angles for each layer
        self.params = nn.Parameter(torch.randn(n_layers, n_wires))
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x
        for i in range(self.n_wires):
            tq.ry(q_device, wires=[i], params=x[:, i])
        # Variational layers
        for l in range(self.n_layers):
            for i in range(self.n_wires):
                tq.ry(q_device, wires=[i], params=self.params[l, i])
            # Entanglement: CX between consecutive wires
            for i in range(self.n_wires - 1):
                tq.cx(q_device, wires=[i, i + 1])
        # Encode y with negative parameters
        for i in range(self.n_wires):
            tq.ry(q_device, wires=[i], params=-y[:, i])

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a variational TorchQuantum ansatz."""
    def __init__(self, n_wires: int = 4, n_layers: int = 1):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(n_wires, n_layers)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
