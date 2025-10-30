"""Quantum kernel construction using TorchQuantum ansatz."""
from __future__ import annotations

from typing import Sequence, List, Dict, Any

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import torch.nn as nn

__all__ = [
    "QuantumKernelMethod",
    "KernelFactory",
    "kernel_matrix",
]

class QuantumKernelMethod(tq.QuantumModule):
    """Variational quantum kernel with a fixed ansatz that can be swapped
    for a learnable classical RBF kernel via the factory.
    """
    def __init__(self, n_wires: int = 4, ansatz: List[Dict[str, Any]] | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = ansatz or [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode two feature vectors and compute the overlap."""
        q_device.reset_states(x.shape[0])
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def forward_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper to return the kernel value."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between two lists of tensors using the quantum kernel."""
    kernel = QuantumKernelMethod()
    return np.array([[kernel.forward_kernel(x, y).item() for y in b] for x in a])

class KernelFactory:
    """Factory to instantiate either a classical or quantum kernel."""
    def __init__(self, use_quantum: bool = False, **kwargs: Any) -> None:
        self.use_quantum = use_quantum
        self.kwargs = kwargs

    def get_kernel(self) -> tq.QuantumModule | nn.Module:
        if self.use_quantum:
            return QuantumKernelMethod(**self.kwargs)
        else:
            from. import QuantumKernelMethod as ClassicalKernel
            return ClassicalKernel(**self.kwargs)
