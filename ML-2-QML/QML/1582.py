"""Quantum kernel construction with configurable depth and hybrid support.

This module extends the original quantum kernel by adding a depth‑controlled
ansatz, a fidelity‑based measurement and a hybrid kernel that adds the
classical RBF part.  It exposes a small factory and a ``kernel_matrix``
function compatible with the classical counterpart.
"""

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, Union

class BaseQKernel(tq.QuantumModule):
    """Abstract quantum kernel base."""
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class QuantumRBF(tq.QuantumModule):
    """Quantum RBF kernel with depth‑controlled Ry ansatz."""
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(x.shape[0])
        # encode x
        for _ in range(self.depth):
            for w in range(self.n_wires):
                func_name_dict["ry"](self.q_device, wires=[w], params=x[:, w])
        # encode y with negative sign
        for _ in range(self.depth):
            for w in range(self.n_wires):
                func_name_dict["ry"](self.q_device, wires=[w], params=-y[:, w])
        # fidelity measurement on first qubit
        states = self.q_device.states.view(-1, 2**self.n_wires)
        fidelity = torch.abs(states[:, 0]) ** 2
        return fidelity.view(x.shape[0], 1)

class HybridRBF(tq.QuantumModule):
    """Hybrid kernel: sum of classical RBF and quantum RBF."""
    def __init__(self, gamma: Union[float, torch.Tensor] = 1.0,
                 n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.gamma = torch.tensor(gamma, dtype=torch.float32) if not isinstance(gamma, torch.Tensor) else gamma
        self.n_wires = n_wires
        self.depth = depth
        self.quantum = QuantumRBF(n_wires, depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        classic = torch.exp(-self.gamma * torch.sum((x - y) ** 2, dim=-1, keepdim=True))
        quantum = self.quantum(x, y)
        return classic + quantum

def kernel_matrix(a: Sequence[Union[torch.Tensor, np.ndarray]],
                  b: Sequence[Union[torch.Tensor, np.ndarray]]) -> np.ndarray:
    kernel = QuantumRBF()
    a_t = [torch.as_tensor(ai, dtype=torch.float32) for ai in a]
    b_t = [torch.as_tensor(bi, dtype=torch.float32) for bi in b]
    mat = torch.stack([kernel(x, y) for x in a_t for y in b_t])
    return mat.cpu().numpy().reshape(len(a), len(b))

__all__ = ["QuantumRBF", "HybridRBF", "kernel_matrix"]
