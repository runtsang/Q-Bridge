"""Quantum kernel using a variational circuit with learnable rotation angles."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torch.nn import Parameter

class QuantumKernelMethod(tq.QuantumModule):
    """Variational quantum kernel.

    The circuit consists of:
        1. A layer of Ry rotations with data‑dependent angles.
        2. A CNOT entangling layer between consecutive qubits.
        3. A second layer of Ry rotations with the negative of the second data point.
    The rotation angles are learnable parameters that can be optimised jointly
    with a downstream learning model.  The kernel value is the absolute overlap
    between the two prepared states.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.params = Parameter(torch.randn(n_wires))
        self.cnot_pairs = [(i, i + 1) for i in range(n_wires - 1)]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode data points ``x`` and ``y`` and prepare the overlap state."""
        q_device.reset_states(x.shape[0])
        # First rotation layer
        for i in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=[i], params=x[:, i] + self.params[i])
        # Entangling layer
        for control, target in self.cnot_pairs:
            func_name_dict["cnot"](q_device, wires=[control, target])
        # Second rotation layer with negative y
        for i in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=[i], params=-y[:, i] + self.params[i])

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel value (overlap) for two batches."""
        self.forward(self.q_device, x, y)
        # Overlap with the reference |0⟩ state
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the Gram matrix between two batches ``a`` and ``b``."""
        return self.kernel_value(a, b)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Convenience wrapper that accepts a list of 1‑D tensors and returns a NumPy array."""
    kernel = QuantumKernelMethod()
    return kernel.kernel_value(torch.stack(a), torch.stack(b)).detach().cpu().numpy()

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
