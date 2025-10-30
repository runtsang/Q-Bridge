# Quantum kernel construction using a parameterized feature map.

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict, op_name_dict
from typing import Sequence

__all__ = ["Kernel", "kernel_matrix"]

class KernalAnsatz(tq.QuantumModule):
    """Feature map consisting of a list of quantum operations."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel using a configurable feature map."""
    def __init__(self, n_wires: int = 4, n_layers: int = 2, backend: str = 'cpu'):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.q_device = tq.QuantumDevice(n_wires=n_wires, backend=backend)
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self):
        func_list = []
        for layer in range(self.n_layers):
            for wire in range(self.n_wires):
                func_list.append({"input_idx": [wire], "func": "ry", "wires": [wire]})
            for wire in range(self.n_wires - 1):
                func_list.append({"input_idx": [], "func": "cx", "wires": [wire, wire + 1]})
        return KernalAnsatz(func_list)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        self.q_device.reset_states(batch)
        for info in self.ansatz.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix_np(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        a_t = torch.tensor(a, dtype=torch.float32, device='cpu')
        b_t = torch.tensor(b, dtype=torch.float32, device='cpu')
        n, d = a_t.shape
        m, _ = b_t.shape
        result = torch.empty((n, m))
        for i in range(n):
            for j in range(m):
                result[i, j] = self.forward(a_t[i].unsqueeze(0), b_t[j].unsqueeze(0))
        return result.cpu().numpy()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return kernel.kernel_matrix_np(a, b)
