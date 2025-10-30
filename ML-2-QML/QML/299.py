"""Quantum kernel construction using TorchQuantum ansatz with depth and parameter sharing."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum.functional import op_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Compatibility alias for original KernalAnsatz. Encodes data via a list of gates."""
    def __init__(self, func_list: Sequence[dict]) -> None:
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
    """Quantum RBF‑like kernel with variational ansatz, depth control and parameter sharing."""
    def __init__(self, n_wires: int = 4, depth: int = 2, shared_params: bool = True) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.shared_params = shared_params
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self) -> KernalAnsatz:
        """Construct a parameter‑shared or independent ansatz."""
        func_list = []
        for d in range(self.depth):
            # Parameterized rotation on each wire
            for w in range(self.n_wires):
                func_list.append({
                    "input_idx": [w],
                    "func": "ry",
                    "wires": [w]
                })
            # Entangling layer
            for w in range(self.n_wires - 1):
                func_list.append({
                    "input_idx": [],
                    "func": "cx",
                    "wires": [w, w + 1]
                })
        return KernalAnsatz(func_list)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel value between two classical vectors using variational ansatz."""
        # Handle batched inputs
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        # Reset device and encode x
        self.q_device.reset_states(x.shape[0])
        self.ansatz(self.q_device, x, y)
        # Return absolute overlap of first amplitude
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between two sequences of tensors."""
        result = []
        for x in a:
            row = []
            for y in b:
                val = self(x, y).item()
                row.append(val)
            result.append(row)
        return np.array(result)


__all__ = ["KernalAnsatz", "Kernel"]
