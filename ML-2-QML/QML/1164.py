"""Quantum kernel module with a variational ansatz and batched evaluation.

The module implements a hybrid kernel that combines a classical RBF
component with a quantum feature map.  The quantum part uses a
parameter‑efficient ansatz that can be easily extended.  The API
mirrors the classical implementation for seamless side‑by‑side
experiments.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class KernalAnsatz(tq.QuantumModule):
    """Quantum feature map based on a shallow Ry‑entanglement circuit.

    The ansatz is parameter‑driven and supports batched encoding of
    classical data.  It can be extended with additional gates or
    entangling layers without changing the public API.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.func_list = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Hybrid quantum‑classical kernel.

    It evaluates a quantum feature map and optionally a classical RBF
    component.  The two kernels are combined linearly with a tunable
    weight `alpha`.
    """
    def __init__(
        self,
        n_wires: int = 4,
        use_classical: bool = True,
        gamma: float = 1.0,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(n_wires=self.n_wires)
        self.use_classical = use_classical
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Quantum part
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        quantum = torch.abs(self.q_device.states.view(-1)[0])

        if self.use_classical:
            diff = x - y
            classical = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
            return self.alpha * classical + (1.0 - self.alpha) * quantum
        return quantum

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int = 4, use_classical: bool = True, gamma: float = 1.0, alpha: float = 0.5) -> np.ndarray:
    """Compute a Gram matrix for two lists of 1‑D tensors using the hybrid kernel."""
    a = torch.stack(a)
    b = torch.stack(b)
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    classical = torch.exp(-gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    kernel = Kernel(n_wires=n_wires, use_classical=use_classical, gamma=gamma, alpha=alpha)
    quantum_vals = []
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            kernel.ansatz(kernel.q_device, a[i].unsqueeze(0), b[j].unsqueeze(0))
            quantum_vals.append(torch.abs(kernel.q_device.states.view(-1)[0]))
    quantum = torch.tensor(quantum_vals).reshape(a.shape[0], b.shape[0])
    return (alpha * classical + (1.0 - alpha) * quantum).cpu().numpy()

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
