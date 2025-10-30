"""Hybrid quantum kernel with self‑attention weighting.

The quantum kernel is evaluated by encoding two samples via a fixed
parameterised circuit and measuring the overlap of the resulting states.
A second, attention‑style circuit is applied to produce a scalar weight
derived from the expectation value of a Pauli‑Z operator.  The final
kernel value is the product of the overlap and the attention weight.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum import operators as ops
from typing import Sequence

class HybridKernelAttention(tq.QuantumModule):
    """
    Hybrid quantum kernel that multiplies a quantum‑kernel overlap with a
    quantum‑derived attention weight.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Parameters that emulate the rotation and entanglement
        # of the quantum self‑attention block
        self.rotation_params = torch.nn.Parameter(
            torch.randn(n_wires * 3) * 0.1)
        self.entangle_params = torch.nn.Parameter(
            torch.randn(n_wires - 1) * 0.1)

    def _kernel_overlap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Encode x and y with a simple ry‑ansatz and return the absolute
        overlap of the resulting states.
        """
        self.q_device.reset_states(x.shape[0])

        # Encode x
        for i in range(self.n_wires):
            func_name_dict["ry"](self.q_device, wires=i, params=x[:, i])
        # Encode y
        for i in range(self.n_wires):
            func_name_dict["ry"](self.q_device, wires=i, params=y[:, i])

        # Overlap = |⟨ψ_x|ψ_y⟩|
        return torch.abs(self.q_device.states.view(-1)[0])

    def _attention_weight(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Encode x and y with a rotation‑entanglement circuit and return the
        absolute expectation value of Z on the first qubit as a weight.
        """
        self.q_device.reset_states(x.shape[0])

        # Rotation gates (fixed parameters, independent of data)
        for i in range(self.n_wires):
            func_name_dict["rx"](self.q_device,
                                 wires=i,
                                 params=self.rotation_params[i, 0])
            func_name_dict["ry"](self.q_device,
                                 wires=i,
                                 params=self.rotation_params[i, 1])
            func_name_dict["rz"](self.q_device,
                                 wires=i,
                                 params=self.rotation_params[i, 2])

        # Entangling gates
        for i in range(self.n_wires - 1):
            func_name_dict["crx"](self.q_device,
                                  wires=[i, i + 1],
                                  params=self.entangle_params[i])

        # Expectation of Z on the first qubit
        return torch.abs(self.q_device.expectation(ops.z, wires=0))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return the hybrid quantum kernel value for a single pair.
        """
        overlap = self._kernel_overlap(x, y)
        attn = self._attention_weight(x, y)
        return overlap * attn

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute the Gram matrix for two collections of samples using the
    hybrid quantum‑attention kernel.
    """
    kernel = HybridKernelAttention()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernelAttention", "kernel_matrix"]
