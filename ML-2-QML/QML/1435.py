"""Quantum kernel construction with entanglement and multi‑wire support."""

from __future__ import annotations

from typing import Sequence, List, Dict, Optional

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
]


class KernalAnsatz(tq.QuantumModule):
    """Parameterized ansatz that encodes data via single‑qubit rotations and optional entangling gates."""

    def __init__(self, data_gates: List[Dict], entanglement_gates: Optional[List[Dict]] = None) -> None:
        super().__init__()
        self.data_gates = data_gates
        self.entanglement_gates = entanglement_gates or []

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # encode x
        for g in self.data_gates:
            params = x[:, g["input_idx"]] if tq.op_name_dict[g["func"]].num_params else None
            func_name_dict[g["func"]](q_device, wires=g["wires"], params=params)
        # entanglement
        for g in self.entanglement_gates:
            func_name_dict[g["func"]](q_device, wires=g["wires"])
        # encode y negatively
        for g in reversed(self.data_gates):
            params = -y[:, g["input_idx"]] if tq.op_name_dict[g["func"]].num_params else None
            func_name_dict[g["func"]](q_device, wires=g["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel that uses a configurable ansatz."""

    def __init__(
        self,
        n_wires: int = 4,
        data_gates: Optional[List[Dict]] = None,
        entanglement_gates: Optional[List[Dict]] = None,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        if data_gates is None:
            data_gates = [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        self.ansatz = KernalAnsatz(data_gates, entanglement_gates)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    n_wires: int = 4,
    data_gates: Optional[List[Dict]] = None,
    entanglement_gates: Optional[List[Dict]] = None,
) -> np.ndarray:
    kernel = Kernel(
        n_wires=n_wires,
        data_gates=data_gates,
        entanglement_gates=entanglement_gates,
    )
    return np.array([[kernel(x, y).item() for y in b] for x in a])
