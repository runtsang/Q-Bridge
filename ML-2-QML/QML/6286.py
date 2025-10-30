"""Quantum kernel implementation using TorchQuantum.

The kernel encodes two classical feature vectors into quantum
states using a parameterised rotation ansatz and returns the
absolute value of the overlap of the resulting states.
"""

from __future__ import annotations

import typing as t

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuantumFeatureMap(tq.QuantumModule):
    """Parameterized quantum feature map that encodes data into a state."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        """Encode a batch of classical vectors x."""
        q_device.reset_states(x.shape[0])
        for wire in range(self.n_wires):
            # each qubit receives a rotation around Y by the corresponding feature
            func_name_dict["ry"](q_device, wires=[wire], params=x[:, wire])


class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.feature_map = QuantumFeatureMap(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return kernel value for two feature vectors."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        # Encode x
        self.feature_map(self.q_device, x)

        # Encode y with inverse rotations to compute overlap
        self.feature_map(self.q_device, -y)

        # Overlap is the magnitude of the amplitude of the |0...0> state
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: t.Sequence[torch.Tensor], b: t.Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two sets of feature vectors."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["Kernel", "kernel_matrix"]
