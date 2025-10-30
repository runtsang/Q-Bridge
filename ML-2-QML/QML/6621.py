"""Quantum kernel construction using a variational ansatz with entanglement."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class _VariationalAnsatz(tq.QuantumModule):
    """Internal helper that builds a depth‑controlled variational circuit."""

    def __init__(self, n_wires: int, depth: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.func_list = self._build_func_list()

    def _build_func_list(self) -> list[dict]:
        func_list = []
        for d in range(self.depth):
            # Single‑qubit rotations
            for w in range(self.n_wires):
                func_list.append({"input_idx": [w], "func": "ry", "wires": [w]})
            # Entangling CNOTs in a ring topology
            for w in range(self.n_wires):
                func_list.append({"input_idx": [], "func": "cx", "wires": [w, (w + 1) % self.n_wires]})
        return func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Encode y with negative parameters
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel evaluated via a variational ansatz with tunable depth and entanglement."""

    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = _VariationalAnsatz(n_wires=self.n_wires, depth=self.depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the overlap between two encoded states.

        Parameters
        ----------
        x : torch.Tensor
            Batch of samples of shape (n, d).
        y : torch.Tensor
            Batch of samples of shape (m, d).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (n, m).
        """
        # Ensure matching dimensionality
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int = 4, depth: int = 2) -> np.ndarray:
    """
    Compute the Gram matrix between two sequences of tensors using the
    :class:`QuantumKernelMethod` kernel.

    Parameters
    ----------
    a : Sequence[torch.Tensor]
        First sequence of tensors.
    b : Sequence[torch.Tensor]
        Second sequence of tensors.
    n_wires : int, optional
        Number of qubits in the ansatz.
    depth : int, optional
        Depth of the variational circuit.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    kernel = QuantumKernelMethod(n_wires=n_wires, depth=depth)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
