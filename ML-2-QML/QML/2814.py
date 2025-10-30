"""Quantum kernel with attention‑style ansatz."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class KernalAnsatz(tq.QuantumModule):
    """Quantum attention ansatz: rotations, entangling, inverse rotations."""
    def __init__(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        super().__init__()
        self.rotation_params = rotation_params
        self.entangle_params = entangle_params
        self.n_wires = rotation_params.shape[0] // 3  # assume 3 params per qubit

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # Reset device with batch size
        q_device.reset_states(x.shape[0])
        # Forward rotations for x
        for i in range(self.n_wires):
            params = x[:, i]
            tq.ry(q_device, wires=[i], params=params)
        # Entangling gates (CRX) to mimic attention interactions
        for i in range(self.n_wires - 1):
            tq.crx(q_device, wires=[i, i+1], params=self.entangle_params[i])
        # Reverse rotations for y (negative)
        for i in range(self.n_wires):
            params = -y[:, i]
            tq.ry(q_device, wires=[i], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel using the attention ansatz."""
    def __init__(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> None:
        super().__init__()
        self.n_wires = rotation_params.shape[0] // 3
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(rotation_params, entangle_params)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Overlap of first basis state
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    rotation_params: np.ndarray,
    entangle_params: np.ndarray,
) -> np.ndarray:
    kernel = Kernel(rotation_params, entangle_params)
    return np.array(
        [[kernel(x, y).item() for y in b] for x in a]
    )

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
