"""Quantum kernel using a variational PennyLane circuit."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml
import torch
from torch import nn


class QuantumCircuit:
    """Encodes data with a variational circuit and returns the quantum state."""

    def __init__(self, n_wires: int = 4, dev: qml.Device | None = None) -> None:
        self.n_wires = n_wires
        self.dev = dev or qml.device("default.qubit", wires=n_wires)

    def circuit(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        @qml.qnode(self.dev)
        def inner(x, params):
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i)
            for i in range(self.n_wires):
                qml.RY(params[i], wires=i)
            return qml.state()
        return inner(x, params)

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute kernel as absolute inner product of two states."""
        state_x = self.circuit(x, self.params)
        state_y = self.circuit(y, self.params)
        return np.abs(np.vdot(state_x, state_y))


class QuantumKernelMethod(nn.Module):
    """
    Quantum kernel with trainable parameters.
    Supports batched evaluation and integration with scikit‑learn estimators.
    """

    def __init__(self, n_wires: int = 4, dev_name: str = "default.qubit") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.dev = qml.device(dev_name, wires=n_wires)
        self.params = nn.Parameter(torch.randn(n_wires))
        self.circuit = QuantumCircuit(n_wires, self.dev)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Batch kernel evaluation."""
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        kernels = np.array([self.circuit.kernel(xi, yi) for xi, yi in zip(x_np, y_np)])
        return torch.tensor(kernels, device=x.device, dtype=x.dtype)

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Return Gram matrix between two sequences of tensors."""
        return np.array(
            [[self.circuit.kernel(x.detach().cpu().numpy(), y.detach().cpu().numpy()).item() for y in b] for x in a]
        )


__all__ = ["QuantumKernelMethod", "QuantumCircuit"]
