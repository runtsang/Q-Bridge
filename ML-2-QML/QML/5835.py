"""Quantum kernel construction with parameterized TorchQuantum ansatz and hybrid support."""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import ry

class QuantumKernelMethod(tq.QuantumModule):
    """
    Quantum kernel module that encodes data through a trainable ansatz
    and returns the absolute overlap between the resulting states.

    Parameters
    ----------
    n_wires : int, default=4
        Number of qubits.
    params : Optional[torch.Tensor], default=None
        Flattened trainable parameters for the ansatz. If None, they are
        initialized randomly.
    """

    def __init__(self, n_wires: int = 4, params: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.params = nn.Parameter(
            torch.randn(n_wires) if params is None else params
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel between two batches of data.
        ``x`` and ``y`` are 1‑D tensors of length ``n_wires``.
        """
        self.q_device.reset_states(1)
        # Encode x
        for i in range(self.n_wires):
            ry(self.q_device, i, params=x[i])
        # Apply trainable ansatz
        for i in range(self.n_wires):
            ry(self.q_device, i, params=self.params[i])
        # Encode y with inverse parameters
        for i in range(self.n_wires):
            ry(self.q_device, i, params=-y[i])
        # Compute overlap
        overlap = torch.abs(self.q_device.states.view(-1)[0]) ** 2
        return overlap

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
        module = QuantumKernelMethod(n_wires=len(a[0]))
        a_tensor = torch.stack(a)
        b_tensor = torch.stack(b)
        k = torch.zeros((len(a), len(b)))
        for i, x in enumerate(a_tensor):
            for j, y in enumerate(b_tensor):
                k[i, j] = module(x, y)
        return k.detach().cpu().numpy()

    def hybrid_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        classical_kernel: torch.Tensor,
        mode: str = "product",
    ) -> torch.Tensor:
        """
        Combine a classical kernel matrix with the quantum kernel.
        ``mode`` can be ``product`` or ``sum``.
        """
        qk = self.forward(x, y)
        if mode == "product":
            return qk * classical_kernel
        elif mode == "sum":
            return qk + classical_kernel
        else:
            raise ValueError(f"Unsupported mode {mode}")

__all__ = ["QuantumKernelMethod"]
