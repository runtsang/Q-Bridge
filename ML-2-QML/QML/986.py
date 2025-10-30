"""Quantum kernel construction using TorchQuantum ansatz with parameterized gates."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torch import nn

__all__ = ["QuantumKernelMethod", "kernel_matrix"]

class QuantumKernelMethod(tq.QuantumModule):
    """
    Quantum kernel evaluated via a parameterized ansatz.
    The ansatz includes trainable rotation angles that depend on the input data.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Learnable parameters for Ry rotations
        self.theta = nn.Parameter(torch.randn(self.n_wires))
        # Build ansatz
        self.ansatz = tq.QuantumModule()
        for i in range(self.n_wires):
            self.ansatz.add_layer(tq.RY, wires=[i], params=self.theta[i])
        # Add entangling CNOTs
        for i in range(self.n_wires - 1):
            self.ansatz.add_layer(tq.CNOT, wires=[i, i+1])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel between two batches of vectors.
        x: (B, D)
        y: (C, D)
        Returns: (B, C) kernel matrix.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        B, D = x.shape
        C, _ = y.shape
        kernel_values = torch.empty(B, C)
        for i in range(B):
            for j in range(C):
                self.q_device.reset_states(1)
                # Encode x[i]
                for idx, val in enumerate(x[i]):
                    tq.RY(self.q_device, wires=idx, params=val)
                # Apply ansatz
                self.ansatz(self.q_device)
                # Encode -y[j]
                for idx, val in enumerate(y[j]):
                    tq.RY(self.q_device, wires=idx, params=-val)
                # Apply ansatz again
                self.ansatz(self.q_device)
                kernel_values[i, j] = torch.abs(self.q_device.states.view(-1)[0])
        return kernel_values

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Evaluate the Gram matrix between datasets ``a`` and ``b`` using the quantum kernel.
    """
    kernel = QuantumKernelMethod()
    return np.array([[kernel(x, y).item() for y in b] for x in a])
