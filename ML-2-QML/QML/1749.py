"""Quantum kernel construction using Pennylane with a variational ansatz."""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as npnp
import torch
import numpy as np
from typing import Sequence

class KernalAnsatz:
    """Variational quantum circuit encoding classical data."""
    def __init__(self, data_dim: int, wires: int):
        self.data_dim = data_dim
        self.wires = wires
        self.dev = qml.device("default.qubit", wires=wires)
        self.variational_params = npnp.array([0.0] * (data_dim + wires - 1), requires_grad=True)

    def _circuit(self, x, y):
        # Encode x
        for idx in range(self.data_dim):
            qml.RY(x[idx], wires=idx)
        # Variational layers
        for idx in range(self.wires):
            qml.RY(self.variational_params[idx], wires=idx)
        # Entanglement
        for idx in range(self.wires - 1):
            qml.CNOT(wires=[idx, idx + 1])
        # Encode -y
        for idx in range(self.data_dim):
            qml.RY(-y[idx], wires=idx)
        # Return probability of measuring |0> on first qubit
        return qml.probs(wires=0)[0]

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> npnp.ndarray:
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        return self._circuit(x_np, y_np)

class Kernel:
    """Quantum kernel using the variational ansatz."""
    def __init__(self, data_dim: int = 4, wires: int = 4):
        self.data_dim = data_dim
        self.wires = wires
        self.ansatz = KernalAnsatz(data_dim, wires)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> npnp.ndarray:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        return np.abs(self.ansatz(x, y))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
