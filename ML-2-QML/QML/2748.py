from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

class PatchAnsatz(tq.QuantumModule):
    """Encodes a 4‑qubit patch via a programmable list of gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class HybridQuantumKernelAnsatz(tq.QuantumModule):
    """Quantum kernel that evaluates the product of patch‑wise RY‑based kernels."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = PatchAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return product of kernels over all 2×2 patches."""
        x = x.reshape(1, -1).view(1, 28, 28)
        y = y.reshape(1, -1).view(1, 28, 28)
        kernel_val = torch.ones(1, device=x.device)
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch_x = torch.stack(
                    [x[0, r, c], x[0, r, c + 1], x[0, r + 1, c], x[0, r + 1, c + 1]],
                    dim=0,
                )
                patch_y = torch.stack(
                    [y[0, r, c], y[0, r, c + 1], y[0, r + 1, c], y[0, r + 1, c + 1]],
                    dim=0,
                )
                self.ansatz(self.q_device, patch_x, patch_y)
                patch_kernel = torch.abs(self.q_device.states.view(-1)[0])
                kernel_val *= patch_kernel
        return kernel_val

class Kernel(tq.QuantumModule):
    """Convenience wrapper for :class:`HybridQuantumKernelAnsatz`."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = HybridQuantumKernelAnsatz(self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridQuantumKernelAnsatz", "Kernel", "kernel_matrix"]
