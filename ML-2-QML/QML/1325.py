"""Hybrid quantum kernel module using TorchQuantum.

The implementation builds upon the original quantum‑ansatz but adds
entangling layers to increase expressivity.  The kernel is evaluated as the absolute overlap of the
final state with the initial state after applying the data‑encoding circuit for ``x`` and
the inverse circuit for ``y``.  A vectorised ``kernel_matrix`` routine is provided that
accepts either single tensors or iterables of tensors.  The module is GPU‑ready and
supports chunked evaluation for large datasets.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Optional

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import torch.nn as nn

class HybridKernel(tq.QuantumModule):
    """Quantum kernel with a data‑encoding ansatz and entangling layers.

    Parameters
    ----------
    n_wires : int, default=4
        Number of qubits in the device.
    depth : int, default=2
        Number of entangling layers.
    """

    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Build a parameterised ansatz: RY rotations followed by CNOT chains
        self.forward_gates = []
        self.inverse_gates = []

        for d in range(self.depth):
            # Data encoding
            for w in range(self.n_wires):
                self.forward_gates.append(
                    {"func": "ry", "wires": [w], "data_idx": w}
                )
                self.inverse_gates.append(
                    {"func": "ry", "wires": [w], "data_idx": w, "sign": -1}
                )
            # Entangling CNOT chain
            for w in range(self.n_wires - 1):
                self.forward_gates.append(
                    {"func": "cx", "wires": [w, w + 1]}
                )
                self.inverse_gates.append(
                    {"func": "cx", "wires": [w, w + 1]}
                )

    @tq.static_support
    def _apply_circuit(
        self, q_device: tq.QuantumDevice, data: torch.Tensor, gates: list
    ) -> None:
        """Apply a sequence of gates to encode data."""
        q_device.reset_states(data.shape[0])
        for g in gates:
            if "data_idx" in g:
                param = data[:, g["data_idx"]] * g.get("sign", 1)
                func_name_dict[g["func"]](q_device, wires=g["wires"], params=param)
            else:
                func_name_dict[g["func"]](q_device, wires=g["wires"])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for a pair of data points."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self._apply_circuit(self.q_device, x, self.forward_gates)
        self._apply_circuit(self.q_device, y, self.inverse_gates)
        # Overlap with initial state |0...0>
        return torch.abs(self.q_device.states.view(-1)[0])

    @classmethod
    def kernel_matrix(
        cls,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        *,
        n_wires: int = 4,
        depth: int = 2,
        chunk_size: int = 64,
    ) -> np.ndarray:
        """Compute the Gram matrix between two collections of vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Each element should be a 1‑D tensor of shape ``(d,)``.  The function
            accepts either lists or torch tensors; if a single tensor is passed
            it is treated as a batch of vectors.
        n_wires, depth : int
            Parameters for the ansatz.
        chunk_size : int
            Size of the batch processed at once to keep memory usage bounded.
        """
        if isinstance(a, torch.Tensor):
            a = [a[i] for i in range(a.shape[0])]
        if isinstance(b, torch.Tensor):
            b = [b[i] for i in range(b.shape[0])]

        kernel = cls(n_wires=n_wires, depth=depth)
        n, m = len(a), len(b)
        result = torch.empty((n, m), device="cpu", dtype=torch.float64)

        for i in range(0, n, chunk_size):
            a_chunk = torch.stack(a[i : i + chunk_size], dim=0)
            for j in range(0, m, chunk_size):
                b_chunk = torch.stack(b[j : j + chunk_size], dim=0)
                for ii, xi in enumerate(a_chunk):
                    for jj, yj in enumerate(b_chunk):
                        result[i + ii, j + jj] = kernel(xi, yj)
        return result.numpy()

__all__ = ["HybridKernel", "kernel_matrix"]
