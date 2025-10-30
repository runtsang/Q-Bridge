"""Quantum kernel implementation using TorchQuantum.

The :class:`QuantumKernelMethod` class encodes two inputs via a
layered RX‑RY‑CX variational circuit and returns the absolute overlap
of the resulting quantum states.  The circuit depth and number of
wires are configurable, providing a richer feature space than the
original seed.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict, op_name_dict
from typing import Sequence

__all__ = ["QuantumKernelMethod", "kernel_matrix"]

class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel based on a layered RX‑RY‑CX ansatz.

    Parameters
    ----------
    n_wires : int, default 4
        Number of qubits in the device.
    depth : int, default 2
        Number of repetitions of the RX‑RY‑CX block.
    """
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self):
        """Return a list of gate dictionaries describing the ansatz."""
        func_list = []
        for _ in range(self.depth):
            # RX rotations on all wires
            for w in range(self.n_wires):
                func_list.append({"input_idx": [w], "func": "rx", "wires": [w]})
            # RY rotations on all wires
            for w in range(self.n_wires):
                func_list.append({"input_idx": [w], "func": "ry", "wires": [w]})
            # CX entanglement in a ring topology
            for w in range(self.n_wires):
                func_list.append({"input_idx": [], "func": "cx", "wires": [w, (w + 1) % self.n_wires]})
        return func_list

    @tq.static_support
    def _apply_ansatz(self, q_device: tq.QuantumDevice, data: torch.Tensor, reverse: bool = False) -> None:
        """Apply the ansatz to the quantum device."""
        gates = reversed(self.ansatz) if reverse else self.ansatz
        for info in gates:
            params = data[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the quantum kernel value for two 1‑D tensors.

        The method encodes ``x`` and ``y`` into two separate runs of the
        circuit and returns the absolute overlap of the final states.
        """
        # Encode first input
        self.q_device.reset_states(x.shape[0])
        self._apply_ansatz(self.q_device, x, reverse=False)
        state1 = self.q_device.states.clone()

        # Encode second input
        self.q_device.reset_states(y.shape[0])
        self._apply_ansatz(self.q_device, y, reverse=True)
        state2 = self.q_device.states.clone()

        # Compute absolute overlap
        overlap = torch.abs(torch.einsum("ij,ij->i", state1, state2))
        return overlap.squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix for two sequences of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of 1‑D tensors.  The kernel is evaluated pairwise
        between elements of ``a`` and ``b``.
    Returns
    -------
    np.ndarray
        2‑D array of shape ``(len(a), len(b))`` containing the kernel
        values.
    """
    model = QuantumKernelMethod()
    return np.array([[model(x, y).item() for y in b] for x in a])
