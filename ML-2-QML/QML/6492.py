"""Quantum kernel construction using Pennylane with a variational ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import pennylane as qml


class KernalAnsatz(nn.Module):
    """Variational quantum kernel that encodes two classical vectors into a single qubit expectation."""
    def __init__(
        self,
        n_wires: int = 4,
        n_layers: int = 2,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device(device or "default.qubit", wires=n_wires)
        # Trainable parameters: a matrix of shape (n_layers, n_wires)
        self.params = nn.Parameter(torch.randn(n_layers, n_wires))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute value of the overlap between |ψ(x)⟩ and |ψ(y)⟩."""
        # Ensure 1‑D tensors
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x_vec: torch.Tensor, y_vec: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode x
            for i in range(self.n_wires):
                qml.RY(x_vec[i], wires=i)
            # Entangling layer
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Parameterized rotations
            for i in range(self.n_wires):
                qml.RZ(params[0, i], wires=i)
            # Encode y with opposite sign
            for i in range(self.n_wires):
                qml.RY(-y_vec[i], wires=i)
            # Entangling layer
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Parameterized rotations
            for i in range(self.n_wires):
                qml.RZ(params[1, i], wires=i)
            # Return the expectation of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))

        # Compute the kernel value
        val = circuit(x.squeeze(), y.squeeze(), self.params)
        return torch.abs(val)


class Kernel(nn.Module):
    """Convenience wrapper that exposes a classical API for the quantum kernel."""
    def __init__(
        self,
        n_wires: int = 4,
        n_layers: int = 2,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(n_wires=n_wires, n_layers=n_layers, device=device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        return self.ansatz(x, y)


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    n_wires: int = 4,
    n_layers: int = 2,
    device: str | None = None,
) -> np.ndarray:
    """Compute the Gram matrix between two collections of vectors using the variational quantum kernel."""
    kernel = Kernel(n_wires=n_wires, n_layers=n_layers, device=device)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
