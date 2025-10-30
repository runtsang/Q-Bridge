"""Hybrid quantum kernel method using a parameterized encoder and a variational ansatz."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence


class QuantumFeatureEncoder(tq.QuantumModule):
    """Encodes image data into a quantum state via a random layer and trainable rotations."""

    def __init__(self, n_wires: int = 4, n_ops: int = 30) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        # Trainable rotations for each wire
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
        """Apply the encoding circuit to the quantum device."""
        self.random_layer(qdev)
        self.rx(qdev, wires=0, params=x[:, 0])
        self.ry(qdev, wires=1, params=x[:, 1])
        self.rz(qdev, wires=2, params=x[:, 2])
        # Map the fourth feature to a controlled rotation
        self.rx(qdev, wires=3, params=-x[:, 3])


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated on the encoded feature vectors."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires, bsz=1)
        self.encoder = QuantumFeatureEncoder(n_wires=n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the overlap between states prepared from x and y."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        # Prepare state for x
        self.encoder(self.q_device, x)
        state_x = self.q_device.states.clone()
        # Reset and prepare state for y
        self.q_device.reset_states()
        self.encoder(self.q_device, y)
        state_y = self.q_device.states.clone()
        # Compute absolute overlap
        overlap = torch.abs(torch.sum(state_x.conj() * state_y))
        return overlap


class HybridKernelMethod(tq.QuantumModule):
    """
    Hybrid quantum kernel that first encodes images into quantum states
    and then evaluates an overlap‑based kernel.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.kernel = QuantumKernel(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])


__all__ = ["HybridKernelMethod"]
