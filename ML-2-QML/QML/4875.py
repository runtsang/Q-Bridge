"""HybridKernelMethod – quantum side implementation.

This module implements a variational quantum kernel that mirrors the classical
feature extraction pipeline.  The circuit consists of:

* A feature‑map layer that maps classical inputs to a product of RX rotations.
* A convolution‑like block that applies controlled‑X gates between neighboring qubits.
* An attention‑like block that uses parametrised controlled‑RZ gates.
* Measurement of the overlap between two prepared states to obtain the kernel value.

The quantum kernel is compatible with TorchQuantum and can be swapped with the
classical counterpart in the same interface.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Iterable


class QuantumHybridAnsatz(tq.QuantumModule):
    """
    Variational ansatz that encodes classical data, performs a
    convolution‑like interaction, then an attention‑style coupling.
    """
    def __init__(self,
                 conv_depth: int = 1,
                 attn_depth: int = 1) -> None:
        super().__init__()
        self.conv_depth = conv_depth
        self.attn_depth = attn_depth

    @tq.static_support
    def forward(self,
                q_device: tq.QuantumDevice,
                x: torch.Tensor,
                y: torch.Tensor) -> None:
        # Reset and encode first vector
        q_device.reset_states(x.shape[0])
        for wire, val in enumerate(x.squeeze()):
            func_name_dict["rx"](q_device, wires=[wire], params=[val])

        # Convolution‑like controlled‑X patterns
        for _ in range(self.conv_depth):
            for i in range(q_device.n_wires - 1):
                func_name_dict["crx"](q_device, wires=[i, i + 1], params=[np.pi / 4])

        # Attention‑like controlled‑RZ
        for _ in range(self.attn_depth):
            for i in range(q_device.n_wires - 1):
                func_name_dict["crz"](q_device, wires=[i, i + 1], params=[np.pi / 6])

        # Encode second vector with inverse rotations
        for wire, val in enumerate(y.squeeze()):
            func_name_dict["rx"](q_device, wires=[wire], params=[-val])


class QuantumHybridKernel(tq.QuantumModule):
    """
    Quantum kernel that evaluates the overlap of two states prepared by
    `QuantumHybridAnsatz`.  The kernel is returned as a real scalar.
    """
    def __init__(self,
                 n_wires: int = 4,
                 conv_depth: int = 1,
                 attn_depth: int = 1) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumHybridAnsatz(conv_depth=conv_depth,
                                          attn_depth=attn_depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Overlap amplitude between the first and second prepared states
        return torch.abs(self.q_device.states.view(-1)[0])


def quantum_kernel_matrix(a: Iterable[torch.Tensor],
                          b: Iterable[torch.Tensor],
                          n_wires: int = 4) -> np.ndarray:
    """Return the Gram matrix computed by the quantum hybrid kernel."""
    kernel = QuantumHybridKernel(n_wires=n_wires)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["QuantumHybridAnsatz", "QuantumHybridKernel", "quantum_kernel_matrix"]
