#!/usr/bin/env python
"""Quantum kernel method – quantum implementation.

Implements a generic rotation‑entanglement ansatz that can be
used as a drop‑in replacement for the classical RBF kernel.
"""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class KernalAnsatz(tq.QuantumModule):
    """
    Encodes classical data into a fixed set of rotations followed by
    optional entangling gates.  The ansatz is a list of dictionaries
    describing each operation; this representation mirrors the
    seed but now supports a configurable entanglement chain.
    """

    def __init__(self, n_wires: int, entangle: bool = True):
        super().__init__()
        self.n_wires = n_wires
        self.entangle = entangle
        self.gate_list = self._build_gate_list()

    def _build_gate_list(self):
        gates = []
        for i in range(self.n_wires):
            gates.append({"input_idx": [i], "func": "ry", "wires": [i]})
        if self.entangle:
            for i in range(self.n_wires - 1):
                gates.append({"input_idx": [], "func": "cnot", "wires": [i, i + 1]})
        return gates

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for gate in self.gate_list:
            if gate["func"] == "cnot":
                func_name_dict[gate["func"]](q_device, wires=gate["wires"])
            else:
                params = x[:, gate["input_idx"]] if tq.op_name_dict[gate["func"]].num_params else None
                func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=params)
        for gate in reversed(self.gate_list):
            if gate["func"] == "cnot":
                func_name_dict[gate["func"]](q_device, wires=gate["wires"])
            else:
                params = -y[:, gate["input_idx"]]
                func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=params)

class QuantumKernelMethod(tq.QuantumModule):
    """
    Quantum kernel wrapper that exposes the same API as the classical
    counterpart.  The kernel is computed as the overlap between
    states prepared from two samples.
    """

    def __init__(self, n_wires: int = 4, entangle: bool = True, backend: str = "default"):
        super().__init__()
        self.n_wires = n_wires
        self.ansatz = KernalAnsatz(n_wires, entangle)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires, backend=backend)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Optional[Sequence[torch.Tensor]] = None) -> np.ndarray:
        if b is None:
            b = a
        kernel = self
        return np.array([[kernel(x, y).item() for y in b] for x in a])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_wires={self.n_wires}, entangle={self.ansatz.entangle})"

__all__ = ["QuantumKernelMethod"]
