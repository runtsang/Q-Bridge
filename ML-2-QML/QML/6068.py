"""Quantum kernel with a simple Ry‑CNOT ansatz and trainable bandwidth."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumRBFKernel(tq.QuantumModule):
    """Quantum RBF kernel that encodes data with a Ry‑CNOT circuit."""
    def __init__(self, n_wires: int = 4, gamma: float = 1.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        # Quantum device; we reuse the same device for all evaluations
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute |⟨ψ(x)|ψ(y)⟩| where |ψ(z)⟩ is prepared by encoding z
        into a Ry‑CNOT circuit.  The bandwidth γ scales the rotation angles.
        x, y: (batch, n_wires)
        """
        # Encode x
        self.q_device.reset_states(x.shape[0])
        for w in range(self.n_wires):
            func_name_dict["ry"](self.q_device, wires=[w], params=x[:, w] * self.gamma)
        for w in range(self.n_wires - 1):
            func_name_dict["cnot"](self.q_device, wires=[w, w + 1])

        # Encode y with negative angles to compute overlap
        for w in range(self.n_wires):
            func_name_dict["ry"](self.q_device, wires=[w], params=-y[:, w] * self.gamma)
        for w in range(self.n_wires - 1):
            func_name_dict["cnot"](self.q_device, wires=[w, w + 1])

        # The overlap is the absolute value of the amplitude of |0…0⟩
        overlap = torch.abs(self.q_device.states.view(-1)[0])
        return overlap

def kernel_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """
    Compute the Gram matrix using the quantum kernel.
    a: (n, d)
    b: (m, d)
    """
    kernel = QuantumRBFKernel()
    K = torch.zeros(a.shape[0], b.shape[0], dtype=torch.float32)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            K[i, j] = kernel(a[i:i+1], b[j:j+1])
    return K.detach().cpu().numpy()

__all__ = ["QuantumRBFKernel", "kernel_matrix"]
