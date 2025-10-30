"""Quantum kernel using TorchQuantum with a trainable variational ansatz.

The module implements a parameterised circuit that maps classical data onto
a Hilbert space.  The kernel value is the absolute overlap of the resulting
states.  Parameters are trainable, enabling hybrid learning pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict, op_name_dict
from typing import Sequence, List, Dict

__all__ = ["QuantumKernelMethod", "kernel_matrix"]


class QuantumKernelMethod(tq.QuantumModule):
    """Parameterized quantum kernel.

    Parameters
    ----------
    n_wires : int, default=4
        Number of qubits in the device.
    ansatz : list[dict], optional
        List describing the variational circuit. Each dict must contain
        ``func`` (gate name), ``wires`` (list of wires), and
        ``input_idx`` (indices into the input vector that supply the gate
        parameters).  If omitted a default Ry‑only circuit is used.
    """

    def __init__(
        self,
        n_wires: int = 4,
        ansatz: List[Dict[str, object]] | None = None,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        if ansatz is None:
            ansatz = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        self.ansatz = ansatz

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for two data vectors."""
        # Encode x
        for info in self.ansatz:
            params = (
                x[:, info["input_idx"]]
                if op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        # Apply inverse of y
        for info in reversed(self.ansatz):
            params = (
                -y[:, info["input_idx"]]
                if op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
        # Overlap
        return torch.abs(self.q_device.states.view(-1)[0])

    @torch.no_grad()
    def gram_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Compute the kernel Gram matrix for two batches of vectors."""
        a_stack = torch.stack(a)
        b_stack = torch.stack(b)
        mat = torch.empty((len(a), len(b)), device=a_stack.device, dtype=a_stack.dtype)
        for i, x in enumerate(a_stack):
            for j, y in enumerate(b_stack):
                mat[i, j] = self.forward(x, y)
        return mat.numpy()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Convenience wrapper that constructs a default kernel."""
    model = QuantumKernelMethod()
    return model.gram_matrix(a, b)
