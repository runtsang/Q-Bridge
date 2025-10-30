"""Quantum kernel construction with a learnable variational ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]

class KernalAnsatz(tq.QuantumModule):
    """Variational ansatz with trainable parameters.

    The ansatz encodes two classical vectors x and y by applying
    parameterised rotations and a fixed entangling layer.  The
    parameters are optimised jointly with a downstream kernel
    learning objective.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Trainable parameters for each layer and each wire
        self.params = nn.Parameter(torch.randn(n_layers, n_wires, 1))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode x, apply variational circuit, then encode -y."""
        q_device.reset_states(x.shape[0])
        # Encode x
        for wire in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=[wire], params=x[:, wire:wire+1])
        # Variational layers
        for l in range(self.n_layers):
            for wire in range(self.n_wires):
                func_name_dict["ry"](q_device, wires=[wire], params=self.params[l, wire])
            # Entanglement
            for wire in range(self.n_wires - 1):
                func_name_dict["cx"](q_device, wires=[wire, wire + 1])
        # Encode -y (reverse)
        for wire in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=[wire], params=-y[:, wire:wire+1])

class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""
    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = KernalAnsatz(n_wires=n_wires, n_layers=n_layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        self.ansatz(self.q_device, x, y)
        # Return absolute value of first amplitude (overlap)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two sequences of samples."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])
