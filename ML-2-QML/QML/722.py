"""Quantum kernel with trainable variational ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torch import nn
from torchquantum.functional import func_name_dict, op_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Parameterized data‑encoding and entangling circuit.

    The circuit consists of:

    * Data‑encoding layers using Ry gates on each qubit.
    * A trainable entangling layer of CRX gates with learnable rotation angles.
    * A second data‑encoding layer with negative parameters to implement
      the inner‑product trick.

    Parameters
    ----------
    n_wires : int
        Number of qubits (must match the dimensionality of the input data).
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.theta = nn.Parameter(torch.randn(n_wires))
        self.ansatz = [
            {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)
        ] + [
            {"input_idx": [i], "func": "crx", "wires": [i, (i + 1) % n_wires]}
            for i in range(n_wires)
        ] + [
            {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Encode ``x`` and ``y`` on the same device and compute the overlap.

        The circuit is applied with parameters ``x`` and then with ``-y``.
        """
        q_device.reset_states(x.shape[0])
        for info in self.ansatz[: self.n_wires]:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for i, info in enumerate(self.ansatz[self.n_wires : 2 * self.n_wires]):
            params = self.theta[i : i + 1] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in self.ansatz[2 * self.n_wires :]:
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel that can be trained end‑to‑end.

    The kernel is defined as the absolute value of the overlap between
    two quantum states prepared from data points ``x`` and ``y``.
    The circuit is a variational ansatz with trainable entangling gates.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value for a pair of data points.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of shape ``(n_wires,)``.  They will be reshaped to
            ``(1, n_wires)`` internally.

        Returns
        -------
        torch.Tensor
            Scalar kernel value.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two sequences of 1‑D tensors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of tensors of shape ``(n_wires,)``.

        Returns
        -------
        np.ndarray
            NumPy array of shape ``(len(a), len(b))``.
        """
        a_tensor = torch.stack([t for t in a])
        b_tensor = torch.stack([t for t in b])
        with torch.no_grad():
            return np.array([[self.forward(a_i, b_j).item() for b_j in b_tensor] for a_i in a_tensor])


__all__ = ["Kernel"]
