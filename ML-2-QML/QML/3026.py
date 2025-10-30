"""Hybrid quantum kernel embedding a self‑attention inspired ansatz."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class HybridKernelMethod(tq.QuantumModule):
    """
    Variational quantum kernel that mixes data‑relevant rotations with
    an entangling pattern derived from a classical self‑attention graph.
    It mirrors the classical attention mechanism in the circuit layout.
    """

    def __init__(self, n_wires: int = 4, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=n_wires)

        # Build a rotation‑entanglement ansatz that mimics attention connectivity.
        self.ansatz = []
        for d in range(depth):
            for i in range(n_wires):
                self.ansatz.append(
                    {"input_idx": [i], "func": "ry", "wires": [i], "depth": d}
                )
            for i in range(n_wires - 1):
                self.ansatz.append(
                    {"input_idx": [i], "func": "crx", "wires": [i, i + 1], "depth": d}
                )

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Encode two feature vectors into the same device and compute their overlap.
        """
        q_device.reset_states(x.shape[0])

        # Encode x
        for info in self.ansatz:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

        # Uncompute y with inverted parameters
        for info in reversed(self.ansatz):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

        # Return the absolute value of the first amplitude as overlap.
        return torch.abs(q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix via state‑overlap on a batch of inputs.
        """
        return np.array(
            [
                [self.forward(self.q_device, x, y).item() for y in b]
                for x in a
            ]
        )


__all__ = ["HybridKernelMethod"]
