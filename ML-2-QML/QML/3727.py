"""Quantum kernel that embeds classical data via a convolution‑based qubit rotation layer."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum.qdevice import QuantumDevice


class ConvQuantumEncoder(tq.QuantumModule):
    """Maps a 2×2 patch of classical data to rotations on a 4‑qubit block."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = QuantumDevice(n_wires=self.n_wires)
        # base ansatz – simple ry rotations followed by entanglement
        self.ansatz = tq.QuantumModule()
        self.ansatz.add_gate('ry', wires=[0])
        self.ansatz.add_gate('ry', wires=[1])
        self.ansatz.add_gate('ry', wires=[2])
        self.ansatz.add_gate('ry', wires=[3])
        self.ansatz.add_gate('cx', wires=[0, 1])
        self.ansatz.add_gate('cx', wires=[2, 3])

    def forward(self, q_device: QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # x, y are 4‑dim vectors (flattened 2×2 patch)
        q_device.reset_states(x.shape[0])
        # encode x
        for i, wire in enumerate(range(self.n_wires)):
            func_name_dict['ry'](q_device, wires=[wire], params=x[:, i])
        # inverse encoding of y
        for i, wire in enumerate(range(self.n_wires)):
            func_name_dict['ry'](q_device, wires=[wire], params=-y[:, i])


class QuantumKernel(tq.QuantumModule):
    """Kernel that evaluates overlap between two datasets after convolutional encoding."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = QuantumDevice(n_wires=self.n_wires)
        self.encoder = ConvQuantumEncoder()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (batch, 4) flattened patches
        x = x.reshape(-1, 4)
        y = y.reshape(-1, 4)
        self.encoder(self.q_device, x, y)
        # overlap is magnitude of first amplitude
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix using the quantum convolution encoder."""
    kernel = QuantumKernel()
    mat = torch.tensor([[kernel(x, y).item() for y in b] for x in a])
    return mat.numpy()

__all__ = ["ConvQuantumEncoder", "QuantumKernel", "kernel_matrix"]
