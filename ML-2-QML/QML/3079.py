"""Hybrid quantum kernel using QCNN-inspired ansatz.

This module defines :class:`HybridKernelQCNN` that builds a quantum kernel by
encoding data with a convolution‑plus‑pooling ansatz derived from the QCNN
seed.  The kernel is compatible with the classical interface of
``QuantumKernelMethod`` and can be used as a drop‑in replacement.

The implementation uses TorchQuantum and follows the
``Kernel``/``KernalAnsatz`` pattern from the original quantum seed.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class HybridKernalAnsatz(tq.QuantumModule):
    """QCNN‑style ansatz that encodes two data vectors and returns their overlap."""

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Build gate list: conv layer + pool layer
        func_list = []
        # Convolutional block
        func_list += self._conv_layer_gate_list(range(n_wires), 0)
        # Pooling block
        func_list += self._pool_layer_gate_list(range(n_wires), 6)

        self.ansatz = KernalAnsatz(func_list)

    @staticmethod
    def _conv_layer_gate_list(qubits, start_idx):
        func_list = []
        for i in range(0, len(qubits), 2):
            q1, q2 = qubits[i], qubits[i + 1]
            idx = start_idx + i // 2 * 3
            func_list += [
                {"input_idx": [], "func": "rz", "wires": [q2], "params": -np.pi / 2},
                {"input_idx": [], "func": "cx", "wires": [q2, q1]},
                {"input_idx": [idx], "func": "rz", "wires": [q1]},
                {"input_idx": [idx + 1], "func": "ry", "wires": [q2]},
                {"input_idx": [], "func": "cx", "wires": [q1, q2]},
                {"input_idx": [idx + 2], "func": "ry", "wires": [q2]},
                {"input_idx": [], "func": "cx", "wires": [q2, q1]},
                {"input_idx": [], "func": "rz", "wires": [q1], "params": np.pi / 2},
            ]
        return func_list

    @staticmethod
    def _pool_layer_gate_list(qubits, start_idx):
        func_list = []
        for i in range(0, len(qubits), 2):
            q1, q2 = qubits[i], qubits[i + 1]
            idx = start_idx + i // 2 * 3
            func_list += [
                {"input_idx": [], "func": "rz", "wires": [q2], "params": -np.pi / 2},
                {"input_idx": [], "func": "cx", "wires": [q2, q1]},
                {"input_idx": [idx], "func": "rz", "wires": [q1]},
                {"input_idx": [idx + 1], "func": "ry", "wires": [q2]},
                {"input_idx": [], "func": "cx", "wires": [q1, q2]},
                {"input_idx": [idx + 2], "func": "ry", "wires": [q2]},
            ]
        return func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        self.ansatz(q_device, x, y)


class HybridKernelQCNN(tq.QuantumModule):
    """Quantum kernel evaluated via a QCNN‑style ansatz."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = HybridKernalAnsatz(self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix between datasets ``a`` and ``b`` using the QCNN‑style kernel."""
    kernel = HybridKernelQCNN()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["HybridKernelQCNN", "kernel_matrix"]
