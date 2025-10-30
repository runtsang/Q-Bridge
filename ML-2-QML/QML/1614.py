"""Quantum kernel implementation using Pennylane variational ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import pennylane as qml

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]


class KernalAnsatz(nn.Module):
    """Variational circuit that encodes classical data and computes a kernel via state overlap."""

    def __init__(self, num_wires: int = 4, layers: int = 2, seed: int = 42) -> None:
        super().__init__()
        self.num_wires = num_wires
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=num_wires)
        self.seed = seed
        self.params = nn.Parameter(torch.randn(layers, num_wires, 3))

    def encode(self, vec: torch.Tensor, circuit: qml.QNode) -> None:
        """Apply data‑encoding rotations to the circuit."""
        for i in range(self.layers):
            for w in range(self.num_wires):
                circuit.RX(vec[w], wires=w)
            for w in range(self.num_wires - 1):
                circuit.CNOT(wires=[w, w + 1])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value between two input feature vectors."""
        @qml.qnode(self.dev, interface="torch")
        def circuit(vec):
            self.encode(vec, circuit)
            return qml.state()

        state_x = circuit(x)
        state_y = circuit(y)
        overlap = torch.abs(torch.sum(state_x * state_y.conj())) ** 2
        return overlap


class Kernel(nn.Module):
    """Quantum kernel module that wraps :class:`KernalAnsatz`."""

    def __init__(self, num_wires: int = 4, layers: int = 2, seed: int = 42) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(num_wires, layers, seed)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute quantum kernel value."""
        return self.ansatz(x, y)


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    num_wires: int = 4,
    layers: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """Compute Gram matrix between two collections of 1‑D tensors using the quantum kernel."""
    kernel = Kernel(num_wires, layers, seed)
    a = [x.view(-1) for x in a]
    b = [y.view(-1) for y in b]
    return np.array([[kernel(x, y).item() for y in b] for x in a])
